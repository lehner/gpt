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
 
  virtual void execute(PVector<Lattice<T>> &fields, int fast_osites) {

    VECTOR_VIEW_OPEN(fields,fields_v,AcceleratorWrite);

    int n_code = code.size();
    const cgpt_stencil_tensor_code_offload_t* p_code = &code[0];

    typedef decltype(coalescedRead(fields_v[0][0])) obj_t;
    typedef decltype(coalescedReadElement(fields_v[0][0],0)) element_t;
    typedef typename obj_t::scalar_type coeff_t;

    int nd = fields[0].Grid()->Nd();

    int _npb = n_code_parallel_blocks;
    int _npbs = n_code_parallel_block_size;

    uint64_t osites = fields[0].Grid()->oSites();

    int _fast_osites = fast_osites;
    
    if (local) {

      ERR("Not implemented yet");
      
    } else {

      CGPT_CARTESIAN_STENCIL_HALO_EXCHANGE(T,);
      
      // now loop
      uint64_t osites = fields[0].Grid()->oSites();

      ASSERT(_npb == 1);

#define BLOCK_SIZE 16

      // TODO: not-divisible by block_size fix
      ASSERT(osites % BLOCK_SIZE == 0);

      
      accelerator_for(ss_block,osites * _npb / BLOCK_SIZE,T::Nsimd(),{

          uint64_t ss, oblock;

	  MAP_INDEXING(ss, oblock);

#define NN (sizeof(obj_t) / sizeof(element_t))

	  for (int iblock=0;iblock<_npbs;iblock++) {
	    
	    int i = oblock * _npbs + iblock;

	    const auto _p = &p_code[i];
	    const auto _f0 = &_p->factor[0];
	    const auto _f1 = &_p->factor[1];

	    element_t* e_a = (element_t*)&fields_v[_f0->index][BLOCK_SIZE * ss];
	    element_t* e_b = (element_t*)&fields_v[_f1->index][BLOCK_SIZE * ss];
	    element_t* e_c = (element_t*)&fields_v[_p->target][BLOCK_SIZE * ss];

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
	      e_c[NN * ff + _p->element] signature functor(e_a[NN * ff +_f0->element]) * e_b[NN * ff + _f1->element];

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
		e_c[NN * ff + _p->element] *= ((coeff_t)_p->weight);
	      break;
	    }
	  }
	  
	});

      // and cleanup
      CGPT_CARTESIAN_STENCIL_CLEANUP(T,);
      
    }

    VECTOR_VIEW_CLOSE(fields_v);
  }

  virtual void execute(const std::vector<cgpt_Lattice_base*>& __fields, int fast_osites) {
    PVector<Lattice<T>> fields;
    cgpt_basis_fill(fields,__fields);
    execute(fields, fast_osites);
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

  //
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
