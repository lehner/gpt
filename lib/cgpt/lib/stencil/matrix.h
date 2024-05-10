/*
    GPT - Grid Python Toolkit
    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2023  Mattia Bruno

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
  
struct cgpt_stencil_matrix_factor_t {
  int index; // index of field
  int point; // index of shift
  int adj; // adjoint of matrix
};

struct cgpt_stencil_matrix_code_offload_t {
  int target;
  int accumulate;
  ComplexD weight;
  int size;
  cgpt_stencil_matrix_factor_t* factor;
};

struct cgpt_stencil_matrix_code_t {
  int target; // target field index
  int accumulate; // field index to accumulate or -1
  ComplexD weight; // weight of term
  std::vector<cgpt_stencil_matrix_factor_t> factor; 
};

class cgpt_stencil_matrix_base {
 public:
  virtual ~cgpt_stencil_matrix_base() {};
  virtual void execute(const std::vector<cgpt_Lattice_base*>& fields, int fast_osites) = 0;
};

template<typename T>
class cgpt_stencil_matrix : public cgpt_stencil_matrix_base {
 public:

  typedef CartesianStencil<T, T, SimpleStencilParams> CartesianStencil_t;
  typedef CartesianStencilView<T, T, SimpleStencilParams> CartesianStencilView_t;
  
  Vector<cgpt_stencil_matrix_code_offload_t> code;
  Vector<cgpt_stencil_matrix_factor_t> factors;
    
  int n_code_parallel_block_size, n_code_parallel_blocks;
  int local;

  // local == true
  GeneralLocalStencil* general_local_stencil;

  // local == false
  SimpleCompressor<T>* compressor;
  CartesianStencilManager<CartesianStencil_t>* sm;
  
  cgpt_stencil_matrix(GridBase* grid,
		      const std::vector<Coordinate>& shifts,
		      const std::vector<cgpt_stencil_matrix_code_t>& _code,
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
      code[i].accumulate = _code[i].accumulate;
      code[i].weight = _code[i].weight;
      code[i].size = (int)_code[i].factor.size();
      code[i].factor = &factors[nfactors];
      memcpy(code[i].factor, &_code[i].factor[0], sizeof(cgpt_stencil_matrix_factor_t) * code[i].size);
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
      }      

      compressor = new SimpleCompressor<T>();
    }


  }

  virtual ~cgpt_stencil_matrix() {
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
    const cgpt_stencil_matrix_code_offload_t* p_code = &code[0];

    typedef decltype(coalescedRead(fields_v[0][0])) obj_t;

    int nd = fields[0].Grid()->Nd();

    int _npb = n_code_parallel_blocks;
    int _npbs = n_code_parallel_block_size;

    uint64_t osites = fields[0].Grid()->oSites();

    int _fast_osites = fast_osites;
    
    if (local) {

      auto sview = general_local_stencil->View();
      
      accelerator_for(ss_block,osites * _npb,T::Nsimd(),{
	  
          uint64_t ss, oblock;
					      
	  MAP_INDEXING(ss, oblock);
	  
	  for (int iblock=0;iblock<_npbs;iblock++) {
	    
	    int i = oblock * _npbs + iblock;
	    
	    obj_t t;
	    
	    const auto _f0 = &p_code[i].factor[0];
	    fetch(t, _f0->point, ss, fields_v[_f0->index], _f0->adj);
	    
	    for (int j=1;j<p_code[i].size;j++) {
	      obj_t f;
	      const auto _f = &p_code[i].factor[j];
	      fetch(f, _f->point, ss, fields_v[_f->index], _f->adj);
	      t = t * f;
	    }
	    
	    obj_t r = p_code[i].weight * t;
	    if (p_code[i].accumulate != -1)
	      r += coalescedRead(fields_v[p_code[i].accumulate][ss]);
	    coalescedWrite(fields_v[p_code[i].target][ss], r);
	  }
	  
	});
      
    } else {

      CGPT_CARTESIAN_STENCIL_HALO_EXCHANGE(T,);
      
      // now loop
      accelerator_for(ss_block,fields[0].Grid()->oSites() * _npb,T::Nsimd(),{

          uint64_t ss, oblock;
					      
	  MAP_INDEXING(ss, oblock);
	  
	  for (int iblock=0;iblock<_npbs;iblock++) {
	    
	    int i = oblock * _npbs + iblock;
	    
	    obj_t t;
	    
	    const auto _f0 = &p_code[i].factor[0];
	    fetch_cs(stencil_map[_f0->index], t, _f0->point, ss, fields_v[_f0->index], _f0->adj,);
	    
	    for (int j=1;j<p_code[i].size;j++) {
	      obj_t f;
	      const auto _f = &p_code[i].factor[j];
	      fetch_cs(stencil_map[_f->index], f, _f->point, ss, fields_v[_f->index], _f->adj,);
	      t = t * f;
	    }
	    
	    obj_t r = p_code[i].weight * t;
	    if (p_code[i].accumulate != -1)
	      r += coalescedRead(fields_v[p_code[i].accumulate][ss]);
	    coalescedWrite(fields_v[p_code[i].target][ss], r);
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

static void cgpt_convert(PyObject* in, cgpt_stencil_matrix_factor_t& out) {
  ASSERT(PyTuple_Check(in));
  ASSERT(PyTuple_Size(in) == 3);
  cgpt_convert(PyTuple_GetItem(in, 0), out.index);
  cgpt_convert(PyTuple_GetItem(in, 1), out.point);
  cgpt_convert(PyTuple_GetItem(in, 2), out.adj);
}

static void cgpt_convert(PyObject* in, cgpt_stencil_matrix_code_t& out) {
  ASSERT(PyDict_Check(in));

  out.target = get_int(in, "target");
  out.accumulate = get_int(in, "accumulate");
  out.weight = get_complex(in, "weight");

  cgpt_convert(get_key(in, "factor"), out.factor);
}

// not implemented message
template<typename T>
NotEnableIf<isEndomorphism<T>,cgpt_stencil_matrix_base*>
cgpt_stencil_matrix_create(GridBase* grid, PyObject* _shifts,
			   PyObject* _code, long code_parallel_block_size, long local) {
  ERR("cgpt_stencil_matrix not implemented for type %s",typeid(T).name());
}

// implemented for endomorphisms
template<typename T>
EnableIf<isEndomorphism<T>,cgpt_stencil_matrix_base*>
cgpt_stencil_matrix_create(GridBase* grid, PyObject* _shifts,
			   PyObject* _code, long code_parallel_block_size, long local) {

  std::vector<Coordinate> shifts;
  cgpt_convert(_shifts,shifts);

  std::vector<cgpt_stencil_matrix_code_t> code;
  cgpt_convert(_code,code);

  return new cgpt_stencil_matrix<T>(grid,shifts,code,code_parallel_block_size, local);
}
