/*
    GPT - Grid Python Toolkit
    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

struct cgpt_stencil_tensor_code_segment_t {
  int block_size;
  int number_of_blocks;
};

struct cgpt_stencil_tensor_execute_params_t {
  int osites_per_instruction;
  int osites_per_cache_block;
};

struct cgpt_stencil_tensor_factor_t {
  void* base_ptr;
  uint64_t stride;
  int16_t index; // index of field
  int16_t point; // index of shift
  uint16_t element; // index of tensor element
  int16_t is_temporary;
};

struct cgpt_stencil_tensor_code_offload_t {
  void* base_ptr;
  uint64_t stride;
  int16_t target;
  int16_t is_temporary;
  uint16_t element;
  int16_t instruction;
  ComplexD weight;
  uint32_t size;
  cgpt_stencil_tensor_factor_t* factor;
};

struct cgpt_stencil_tensor_code_t {
  int target; // target field index
  int is_temporary;
  int element;
  int instruction;
  ComplexD weight; // weight of term
  std::vector<cgpt_stencil_tensor_factor_t> factor;
};

class cgpt_stencil_tensor_base {
 public:
  virtual ~cgpt_stencil_tensor_base() {};
  virtual void execute(const std::vector<cgpt_Lattice_base*>& fields,
		       const cgpt_stencil_tensor_execute_params_t& params) = 0;
};

template<typename T>
class cgpt_stencil_tensor : public cgpt_stencil_tensor_base {
 public:

  typedef CartesianStencil<T, T, SimpleStencilParams> CartesianStencil_t;
  typedef CartesianStencilView<T, T, SimpleStencilParams> CartesianStencilView_t;
  
  Vector<cgpt_stencil_tensor_code_offload_t> code;
  Vector<cgpt_stencil_tensor_factor_t> factors;

  std::vector<cgpt_stencil_tensor_code_segment_t> segments;
  int local;

  // local == true
  GeneralLocalStencil* general_local_stencil;

  // local == false
  SimpleCompressor<T>* compressor;
  CartesianStencilManager<CartesianStencil_t>* sm;
  
  cgpt_stencil_tensor(GridBase* grid,
		      const std::vector<Coordinate>& shifts,
		      const std::vector<cgpt_stencil_tensor_code_t>& _code,
		      const std::vector<cgpt_stencil_tensor_code_segment_t>& _segments,
		      int _local) :
    code(_code.size()), local(_local), segments(_segments) {

    // test
    size_t code_expected_size = 0;
    for (auto & s : segments)
      code_expected_size += s.block_size * s.number_of_blocks;
    ASSERT(_code.size() == code_expected_size);
    
    // total number of factors
    int nfactors = 0;
    for (int i=0;i<_code.size();i++)
      nfactors += (int)_code[i].factor.size();
    factors.resize(nfactors);
    // fill in code and factors and link them
    nfactors = 0;
    for (int i=0;i<_code.size();i++) {
      code[i].target = _code[i].target;
      code[i].is_temporary = _code[i].is_temporary;
      code[i].element = _code[i].element;
      code[i].instruction = _code[i].instruction;
      code[i].weight = _code[i].weight;
      code[i].size = (uint32_t)_code[i].factor.size();
      if (code[i].size == 0)
	ERR("Cannot create empty factor");
      code[i].factor = &factors[nfactors];
      memcpy(code[i].factor, &_code[i].factor[0], sizeof(cgpt_stencil_tensor_factor_t) * code[i].size);
      nfactors += code[i].size;
    }

    if (local) {
      general_local_stencil = new GeneralLocalStencil(grid,shifts);
    } else {

      sm = new CartesianStencilManager<CartesianStencil_t>(grid, shifts);

      // for all factors that require a non-trivial shift, create a separate stencil object
      for (int i=0;i<nfactors;i++) {
	sm->register_point(factors[i].index, factors[i].point);
      }

      sm->create_stencils(true);

      for (int i=0;i<nfactors;i++) {
	factors[i].point = sm->map_point(factors[i].index, factors[i].point);
	if (factors[i].point != -1)
	  ERR("Only site-local tensor stencils implemented so far");
      }      

      compressor = new SimpleCompressor<T>();
    }


  }

  virtual ~cgpt_stencil_tensor() {
    if (local) {
      delete general_local_stencil;
    } else {
      delete compressor;
      delete sm;
    }
  }

  template<int osites_per_instruction>
  void block_execute(const std::vector<cgpt_Lattice_base*>& fields, int osites_per_cache_block) {

#ifndef GRID_HAS_ACCELERATOR
    typedef typename T::vector_type element_t;
#define NSIMD 1
#else
    typedef typename T::scalar_type element_t;
#define NSIMD (sizeof(typename T::vector_type) / sizeof(typename T::scalar_type))
#endif
    typedef typename T::scalar_type coeff_t;

    VECTOR_ELEMENT_VIEW_OPEN(element_t, fields, fields_v, AcceleratorWrite);

    int n_code = code.size();
    cgpt_stencil_tensor_code_offload_t* p_code = &code[0];

    int nd = fields[0]->get_grid()->Nd();

    uint64_t osites = fields[0]->get_grid()->oSites();

    if (local) {

      ERR("Not implemented yet");
      
    } else {

      //CGPT_CARTESIAN_STENCIL_HALO_EXCHANGE(T,);

#define TC_MOV 0
#define TC_INC 1
#define TC_MOV_NEG 2
#define TC_DEC 3
#define TC_MOV_CC 4
#define TC_INC_CC 5
#define TC_MOV_NEG_CC 6
#define TC_DEC_CC 7
#define TC_MUL 8
#define TC_ADD 9
#define ID(a) a
#define CONJ(a) adj(a)
	      
#define EXECUTE(KB, NN)							\
	      switch (_p->instruction)					\
		{							\
		case TC_INC: KB(+=,*,ID,NN); break;			\
		case TC_MOV: KB(=,*,ID,NN); break;			\
		case TC_DEC: KB(-=,*,ID,NN); break;			\
		case TC_MOV_NEG: KB(=-,*,ID,NN); break;			\
		case TC_INC_CC: KB(+=,*,CONJ,NN); break;		\
		case TC_MOV_CC: KB(=,*,CONJ,NN); break;			\
		case TC_DEC_CC: KB(-=,*,CONJ,NN); break;		\
		case TC_MOV_NEG_CC: KB(=-,*,CONJ,NN); break;		\
		case TC_ADD: KB(=,+,ID,NN); break;			\
		case TC_MUL:						\
		  {							\
		    auto w = ((coeff_t)_p->weight);			\
		    for (int ff=0;ff<NN;ff++)				\
		      e_c[cNN * ff] = w * e_a[aNN * ff];		\
		  }							\
		  break;						\
		}
     
#define KERNEL_BIN(signature, op, functor, NN) {			\
	if (_p->size == 2) {						\
	  auto bNN = _f1->stride;					\
	  element_t* __restrict__ e_b = ((element_t*)_f1->base_ptr) + bNN * NN * MAP_INDEX(_f1,ss) + lane; \
	  for (int ff=0;ff<NN;ff++)					\
	    e_c[cNN * ff] signature functor(e_a[aNN * ff]) op e_b[bNN * ff]; \
	} else {							\
	  for (int ff=0;ff<NN;ff++)					\
	    e_c[cNN * ff] signature functor(e_a[aNN * ff]);		\
	}								\
      }
      

      ASSERT(osites_per_cache_block % osites_per_instruction == 0);

#ifndef GRID_HAS_ACCELERATOR
      thread_region {
#endif
      uint64_t ocache_blocks = (osites + osites_per_cache_block - 1) / osites_per_cache_block;
      for (uint64_t ocache_block = 0;ocache_block < ocache_blocks;ocache_block++) {
	uint64_t osites0 = std::min(ocache_block * osites_per_cache_block, osites);
	uint64_t osites1 = std::min(osites0 + osites_per_cache_block, osites);

	uint64_t osites_in_cache_block = osites1 - osites0;
	
	uint64_t oblocks = osites_in_cache_block / osites_per_instruction;
	uint64_t oblock0 = osites0 / osites_per_instruction;

	uint64_t osites_extra_start = oblocks * osites_per_instruction;
	uint64_t osites_extra = osites_in_cache_block - osites_extra_start;

	// set base_ptr for current views
	for (int i=0;i<n_code;i++) {
	  cgpt_stencil_tensor_code_offload_t* _p = &p_code[i];
	  _p->stride = fields_v_nelements[_p->target] * NSIMD;
	  if (_p->is_temporary) {
	    _p->base_ptr = &fields_v[_p->target][_p->element * NSIMD];
	  } else {
	    _p->base_ptr = &fields_v[_p->target][_p->stride * osites_per_instruction * oblock0 + _p->element * NSIMD];
	  }
	  for (int j=0;j<_p->size;j++) {
	    cgpt_stencil_tensor_factor_t* _f = &_p->factor[j];
	    _f->stride = fields_v_nelements[_f->index] * NSIMD;
	    if (_f->is_temporary) {
	      _f->base_ptr = &fields_v[_f->index][_f->element * NSIMD];
	    } else {
	      _f->base_ptr = &fields_v[_f->index][_f->stride * osites_per_instruction * oblock0 + _f->element * NSIMD];
	    }
	  }
	}
	
	//std::cout << GridLogMessage<< "Group " << osites0 << " to " << osites1 << " has oblocks " << oblocks << " and extra " << osites_extra << " from " << osites_extra_start << " compare to " << osites << std::endl;
	
#ifdef GRID_HAS_ACCELERATOR
#define MAP_INDEX(x,ss) ss
	int coffset = 0;
	for (auto & segment : segments) {
	  int _npb = segment.number_of_blocks;
	  int _npbs = segment.block_size;
	  accelerator_forNB(ss_block, oblocks * _npb, T::Nsimd(), {
	      uint64_t cc = ss_block % _npb;
#else
#define MAP_INDEX(x,ss) ss
	      //(x->is_temporary ? ss : ss)
#define _npb 1
#define _npbs n_code
#define coffset 0
#define cc 0
	  thread_for_in_region(ss_block, oblocks * _npb, {
#endif
      
	      uint64_t ss = ss_block / _npb;
	      
	      for (int ic=0;ic<_npbs;ic++) {

		const cgpt_stencil_tensor_code_offload_t* __restrict__ _p = &p_code[coffset + cc * _npbs + ic];
		const cgpt_stencil_tensor_factor_t* __restrict__ _f0 = &_p->factor[0];
		const cgpt_stencil_tensor_factor_t* __restrict__ _f1 = &_p->factor[1];
		
		int lane = acceleratorSIMTlane(T::Nsimd());
		auto aNN = _f0->stride;
		element_t* __restrict__ e_a = ((element_t*)_f0->base_ptr) + aNN * osites_per_instruction * MAP_INDEX(_f0,ss) + lane;

		auto cNN = _p->stride;
		element_t* __restrict__ e_c = ((element_t*)_p->base_ptr) + cNN * osites_per_instruction * MAP_INDEX(_p,ss) + lane;
		
		EXECUTE(KERNEL_BIN, osites_per_instruction);
	      }
	      
	    });
	  
	  if (osites_extra) {
#ifdef GRID_HAS_ACCELERATOR
	    accelerator_forNB(ss_block, osites_extra * _npb, T::Nsimd(), {
		uint64_t cc = ss_block % _npb;
#else
#define cc 0
            thread_for_in_region(ss_block, osites_extra * _npb, {
#endif
		
		uint64_t ss = ss_block / _npb + osites_extra_start;

		for (int ic=0;ic<_npbs;ic++) {
		  
		  const cgpt_stencil_tensor_code_offload_t* __restrict__ _p = &p_code[coffset + cc * _npbs + ic];
		  const cgpt_stencil_tensor_factor_t* __restrict__ _f0 = &_p->factor[0];
		  const cgpt_stencil_tensor_factor_t* __restrict__ _f1 = &_p->factor[1];
		  
		  int lane = acceleratorSIMTlane(T::Nsimd());
		  auto aNN = _f0->stride;
		  element_t* __restrict__ e_a = ((element_t*)_f0->base_ptr) + aNN * MAP_INDEX(_f0,ss) + lane;
		  
		  auto cNN = _p->stride;
		  element_t* __restrict__ e_c = ((element_t*)_p->base_ptr) + cNN * MAP_INDEX(_p,ss) + lane;

		  EXECUTE(KERNEL_BIN, 1);
		}	    
	      });
	  }

#ifdef GRID_HAS_ACCELERATOR
	  coffset += _npb * _npbs;

          accelerator_barrier();
	}
#endif
      } 
    
      // and cleanup
      //CGPT_CARTESIAN_STENCIL_CLEANUP(T,);
    }

#ifndef GRID_HAS_ACCELERATOR
    }
#endif
    
    VECTOR_ELEMENT_VIEW_CLOSE(fields);
  }

  virtual void execute(const std::vector<cgpt_Lattice_base*>& fields,
		       const cgpt_stencil_tensor_execute_params_t& params) {
    
    switch (params.osites_per_instruction) {
    case 1: block_execute<1>(fields, params.osites_per_cache_block); break;
    case 2: block_execute<2>(fields, params.osites_per_cache_block); break;
    case 4: block_execute<4>(fields, params.osites_per_cache_block); break;
    case 8: block_execute<8>(fields, params.osites_per_cache_block); break;
    case 16: block_execute<16>(fields, params.osites_per_cache_block); break;
    case 32: block_execute<32>(fields, params.osites_per_cache_block); break;
    case 64: block_execute<64>(fields, params.osites_per_cache_block); break;
    case 128: block_execute<128>(fields, params.osites_per_cache_block); break;
    case 256: block_execute<256>(fields, params.osites_per_cache_block); break;
    default: ERR("params.osites_per_instruction = %d not implemented", params.osites_per_instruction);
    }

  }
};

static void cgpt_convert(PyObject* in, cgpt_stencil_tensor_factor_t& out) {
  ASSERT(PyTuple_Check(in));
  ASSERT(PyTuple_Size(in) == 3);
  cgpt_convert(PyTuple_GetItem(in, 0), out.index);
  cgpt_convert(PyTuple_GetItem(in, 1), out.point);
  cgpt_convert(PyTuple_GetItem(in, 2), out.element);

  if (out.index < 0) {
    out.is_temporary = 1;
    out.index = - out.index;
  } else {
    out.is_temporary = 0;
  }
}

static void cgpt_convert(PyObject* in, cgpt_stencil_tensor_code_segment_t& out) {
  ASSERT(PyTuple_Check(in));
  ASSERT(PyTuple_Size(in) == 2);
  cgpt_convert(PyTuple_GetItem(in, 0), out.block_size);
  cgpt_convert(PyTuple_GetItem(in, 1), out.number_of_blocks);
}

static void cgpt_convert(PyObject* in, cgpt_stencil_tensor_code_t& out) {
  ASSERT(PyDict_Check(in));

  out.target = get_int(in, "target");
  if (out.target < 0) {
    out.target = - out.target;
    out.is_temporary = 1;
  } else {
    out.is_temporary = 0;
  }
  out.element = get_int(in, "element");
  out.instruction = get_int(in, "instruction");
  out.weight = get_complex(in, "weight");

  cgpt_convert(get_key(in, "factor"), out.factor);
}

template<typename T>
cgpt_stencil_tensor_base* cgpt_stencil_tensor_create(GridBase* grid, PyObject* _shifts,
						     PyObject* _code, PyObject* _segments,
						     long local) {

  std::vector<Coordinate> shifts;
  cgpt_convert(_shifts,shifts);

  std::vector<cgpt_stencil_tensor_code_t> code;
  cgpt_convert(_code,code);

  std::vector<cgpt_stencil_tensor_code_segment_t> segments;
  cgpt_convert(_segments,segments);

  return new cgpt_stencil_tensor<T>(grid,shifts,code, segments, local);
}
