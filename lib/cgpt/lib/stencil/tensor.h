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

struct cgpt_stencil_tensor_factor_t {
  uint16_t index; // index of field
  int16_t point; // index of shift
  uint16_t element; // index of tensor element
};

struct cgpt_stencil_tensor_code_offload_t {
  uint16_t target;
  uint16_t element;
  int16_t instruction;
  ComplexD weight;
  uint32_t size;
  cgpt_stencil_tensor_factor_t* factor;
};

struct cgpt_stencil_tensor_code_t {
  int target; // target field index
  int element;
  int instruction;
  ComplexD weight; // weight of term
  std::vector<cgpt_stencil_tensor_factor_t> factor;
};

class cgpt_stencil_tensor_base {
 public:
  virtual ~cgpt_stencil_tensor_base() {};
  virtual void execute(const std::vector<cgpt_Lattice_base*>& fields, int fast_osites) = 0;
};

template<typename T>
class cgpt_stencil_tensor : public cgpt_stencil_tensor_base {
 public:

  typedef CartesianStencil<T, T, SimpleStencilParams> CartesianStencil_t;
  typedef CartesianStencilView<T, T, SimpleStencilParams> CartesianStencilView_t;
  
  Vector<cgpt_stencil_tensor_code_offload_t> code;
  Vector<cgpt_stencil_tensor_factor_t> factors;
    
  int n_code_parallel_block_size, n_code_parallel_blocks;
  int local;

  // local == true
  GeneralLocalStencil* general_local_stencil;

  // local == false
  SimpleCompressor<T>* compressor;
  CartesianStencilManager<CartesianStencil_t>* sm;
  
  cgpt_stencil_tensor(GridBase* grid,
		      const std::vector<Coordinate>& shifts,
		      const std::vector<cgpt_stencil_tensor_code_t>& _code,
		      int _n_code_parallel_block_size,
		      int _local) :
    code(_code.size()), local(_local),
    n_code_parallel_block_size(_n_code_parallel_block_size) {

    ASSERT(_code.size() % n_code_parallel_block_size == 0);
    n_code_parallel_blocks = (int)_code.size() / n_code_parallel_block_size;
    
    // total number of factors
    int nfactors = 0;
    for (int i=0;i<_code.size();i++)
      nfactors += (int)_code[i].factor.size();
    factors.resize(nfactors);
    // fill in code and factors and link them
    nfactors = 0;
    for (int i=0;i<_code.size();i++) {
      code[i].target = _code[i].target;
      code[i].element = _code[i].element;
      code[i].instruction = _code[i].instruction;
      code[i].weight = _code[i].weight;
      code[i].size = (int)_code[i].factor.size();
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
  /*

    TODO:
    
    stencils should return different options for current hardware for performance (including max _npb)

  */
  template<int BLOCK_SIZE>
  void block_execute(const std::vector<cgpt_Lattice_base*>& fields, int fast_osites) {

#ifndef GRID_HAS_ACCELERATOR
    typedef typename T::vector_type element_t;
#define NSIMD 1
#else
    typedef typename T::scalar_type element_t;
#define NSIMD (sizeof(typename T::vector_type) / sizeof(typename T::scalar_type))
#endif
    typedef typename element_t::scalar_type coeff_t;
      
    VECTOR_ELEMENT_VIEW_OPEN(element_t, fields, fields_v, AcceleratorWrite);

    int n_code = code.size();
    const cgpt_stencil_tensor_code_offload_t* p_code = &code[0];

    int nd = fields[0]->get_grid()->Nd();

    int _npb = n_code_parallel_blocks;
    int _npbs = n_code_parallel_block_size;

    uint64_t osites = fields[0]->get_grid()->oSites();
    uint64_t osite_blocks = osites;

    int _fast_osites;
    
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

      /*
#ifdef GRID_HAS_ACCELERATOR


      
#define _ID(a) a
#define _CONJ(a) adj(a)
#define _INC(a,b,c)      a + b*c
#define _MOV(a,b,c)          b*c
#define _MOV_NEG(a,b,c)    - b*c
#define _DEC(a,b,c)      a - b*c
#define _MUL(a,b,c)      a*((coeff_t)_p->weight)
#define KERNEL(composition, mod_first)					\
      for (int ff=0;ff < BLOCK_SIZE;ff++)				\
	coalescedWriteElement(fields_v[_p->target][BLOCK_SIZE * ss + ff], \
			      composition(coalescedReadElement(fields_v[_p->target][BLOCK_SIZE * ss + ff], _p->element), \
					  mod_first(coalescedReadElement(fields_v[_f0->index][BLOCK_SIZE * ss + ff], _f0->element)), \
					  coalescedReadElement(fields_v[_f1->index][BLOCK_SIZE * ss + ff], _f1->element)), \
			      _p->element);
      

      osite_blocks = osites / BLOCK_SIZE;

      for (int iter=0;iter<((osites % BLOCK_SIZE == 0) ? 1 : 2);iter++) {

	uint64_t osite_offset = (iter == 0) ? 0 : osite_blocks * BLOCK_SIZE;
	if (iter == 1) {
	  BLOCK_SIZE = 1;
	  osite_blocks = osites - osite_offset;
	}
	
	accelerator_forNB(ss_block,osite_blocks * _npb,T::Nsimd(),{
	    
	    uint64_t ss, oblock;
	    
	    if (_fast_osites) {
	      oblock = ss_block / osite_blocks;
	      ss = osite_offset + ss_block % osite_blocks;
	    } else {
	      ss = osite_offset + ss_block / _npb;
	      oblock = ss_block % _npb;
	    }
	    
	    for (int iblock=0;iblock<_npbs;iblock++) {
	      
	      int i = oblock * _npbs + iblock;
	      
	      const auto _p = &p_code[i];
	      const auto _f0 = &_p->factor[0];
	      const auto _f1 = &_p->factor[1];
	      
	      switch (_p->instruction) {
	      case TC_INC:
		KERNEL(_INC,_ID);
		break;
	      case TC_MOV:
		KERNEL(_MOV,_ID);
		break;
	      case TC_DEC:
		KERNEL(_DEC,_ID);
		break;
	      case TC_MOV_NEG:
		KERNEL(_MOV_NEG,_ID);
		break;
	      case TC_INC_CC:
		KERNEL(_INC,_CONJ);
		break;
	      case TC_MOV_CC:
		KERNEL(_MOV,_CONJ);
		break;
	      case TC_DEC_CC:
		KERNEL(_DEC,_CONJ);
		break;
	      case TC_MOV_NEG_CC:
		KERNEL(_MOV_NEG,_CONJ);
		break;
	      case TC_MUL:
		KERNEL(_MUL,_ID);
		break;
	      }
	    }
	    
	  });
      }
      
      accelerator_barrier();


#else
      */
      
      // CPU version
      ASSERT(osites % BLOCK_SIZE == 0);
      osites /= BLOCK_SIZE;

      int _fast_osites = fast_osites;
      
      accelerator_for(ss_block,osites * _npb,T::Nsimd(),{

          uint64_t ss, oblock;

	  MAP_INDEXING(ss, oblock);

	  for (int iblock=0;iblock<_npbs;iblock++) {
	    
	    int i = oblock * _npbs + iblock;

	    const auto _p = &p_code[i];
	    const auto _f0 = &_p->factor[0];
	    const auto _f1 = &_p->factor[1];

	    int aNN = nelements[_f0->index] * NSIMD;
	    int bNN = nelements[_f1->index] * NSIMD;
	    int cNN = nelements[_p->target] * NSIMD;

	    int lane = acceleratorSIMTlane(T::Nsimd());
	    element_t* __restrict__ e_a = &fields_v[_f0->index][aNN * BLOCK_SIZE * ss + _f0->element * NSIMD + lane];
	    element_t* __restrict__ e_b = &fields_v[_f1->index][bNN * BLOCK_SIZE * ss + _f1->element * NSIMD + lane];
	    element_t* __restrict__ e_c = &fields_v[_p->target][cNN * BLOCK_SIZE * ss + _p->element * NSIMD + lane];

#define TC_MOV 0
#define TC_INC 1
#define TC_MOV_NEG 2
#define TC_DEC 3
#define TC_MOV_CC 4
#define TC_INC_CC 5
#define TC_MOV_NEG_CC 6
#define TC_DEC_CC 7
#define TC_MUL 8
	    
#define ID(a) a
#define CONJ(a) adj(a)
#define KERNEL(signature, functor)					\
	    for (int ff=0;ff<BLOCK_SIZE;ff++)				\
	      e_c[cNN * ff] signature functor(e_a[aNN * ff]) * e_b[bNN * ff];

	    switch (_p->instruction) {
	    case TC_INC:
	      KERNEL(+=,ID);
	      break;
	    case TC_MOV:
	      KERNEL(=,ID);
	      break;
	    case TC_DEC:
	      KERNEL(-=,ID);
	      break;
	    case TC_MOV_NEG:
	      KERNEL(=-,ID);
	      break;
	    case TC_INC_CC:
	      KERNEL(+=,CONJ);
	      break;
	    case TC_MOV_CC:
	      KERNEL(=,CONJ);
	      break;
	    case TC_DEC_CC:
	      KERNEL(-=,CONJ);
	      break;
	    case TC_MOV_NEG_CC:
	      KERNEL(=-,CONJ);
	      break;
	    case TC_MUL:
	      for (int ff=0;ff<BLOCK_SIZE;ff++)	\
		e_c[cNN * ff] *= ((coeff_t)_p->weight);
	      break;
	    }
	  }
	  
	});


      //#endif
      
      // and cleanup
      //CGPT_CARTESIAN_STENCIL_CLEANUP(T,);
      
    }

    VECTOR_ELEMENT_VIEW_CLOSE(fields);
  }

  virtual void execute(const std::vector<cgpt_Lattice_base*>& fields, int kernel_param) {
    
    int _BLOCK_SIZE, _fast_osites;
    if (kernel_param > 0) {
      _BLOCK_SIZE = kernel_param;
      _fast_osites = 1;
    } else {
      _BLOCK_SIZE = -kernel_param;
      _fast_osites = 0;
    }

    switch (_BLOCK_SIZE) {
    case 1: block_execute<1>(fields, _fast_osites); break;
    case 2: block_execute<2>(fields, _fast_osites); break;
    case 4: block_execute<4>(fields, _fast_osites); break;
    case 8: block_execute<8>(fields, _fast_osites); break;
    case 16: block_execute<16>(fields, _fast_osites); break;
    case 32: block_execute<32>(fields, _fast_osites); break;
    case 64: block_execute<64>(fields, _fast_osites); break;
    default: ERR("BLOCK_SIZE = %d not implemented", _BLOCK_SIZE);
    }

  }
};

static void cgpt_convert(PyObject* in, cgpt_stencil_tensor_factor_t& out) {
  ASSERT(PyTuple_Check(in));
  ASSERT(PyTuple_Size(in) == 3);
  cgpt_convert(PyTuple_GetItem(in, 0), out.index);
  cgpt_convert(PyTuple_GetItem(in, 1), out.point);
  cgpt_convert(PyTuple_GetItem(in, 2), out.element);
}

static void cgpt_convert(PyObject* in, cgpt_stencil_tensor_code_t& out) {
  ASSERT(PyDict_Check(in));

  out.target = get_int(in, "target");
  out.element = get_int(in, "element");
  out.instruction = get_int(in, "instruction");
  out.weight = get_complex(in, "weight");

  cgpt_convert(get_key(in, "factor"), out.factor);
}

template<typename T>
cgpt_stencil_tensor_base* cgpt_stencil_tensor_create(GridBase* grid, PyObject* _shifts,
						     PyObject* _code, long code_parallel_block_size,
						     long local) {

  std::vector<Coordinate> shifts;
  cgpt_convert(_shifts,shifts);

  std::vector<cgpt_stencil_tensor_code_t> code;
  cgpt_convert(_code,code);

  return new cgpt_stencil_tensor<T>(grid,shifts,code,code_parallel_block_size, local);
}
