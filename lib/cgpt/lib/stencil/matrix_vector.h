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
  
struct cgpt_stencil_matrix_vector_factor_t {
  int index; // index of field
  int point; // index of shift
  int adj; // adjoint of matrix
};

struct cgpt_stencil_matrix_vector_code_offload_t {
  int target;
  int accumulate;
  int source;
  int source_point;
  ComplexD weight;
  int size;
  cgpt_stencil_matrix_vector_factor_t* factor;
};

struct cgpt_stencil_matrix_vector_code_t {
  int target; // target field index
  int accumulate; // field index to accumulate or -1
  int source; // source field index
  int source_point; // index of source shift
  ComplexD weight; // weight of term
  std::vector<cgpt_stencil_matrix_vector_factor_t> factor; 
};

class cgpt_stencil_matrix_vector_base {
 public:
  virtual ~cgpt_stencil_matrix_vector_base() {};
  virtual void execute(const std::vector<cgpt_Lattice_base*>& matrix_fields, const std::vector<cgpt_Lattice_base*>& vector_fields) = 0;
};

template<typename M, typename V>
class cgpt_stencil_matrix_vector : public cgpt_stencil_matrix_vector_base {
 public:

  GeneralLocalStencil stencil;
  Vector<cgpt_stencil_matrix_vector_code_offload_t> code;
  Vector<cgpt_stencil_matrix_vector_factor_t> factors;
  int n_code_parallel_block_size, n_code_parallel_blocks;
    
  cgpt_stencil_matrix_vector(GridBase* grid,
			     const std::vector<Coordinate>& shifts,
			     const std::vector<cgpt_stencil_matrix_vector_code_t>& _code,
			     int _n_code_parallel_block_size) :
    stencil(grid,shifts), code(_code.size()),
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
      code[i].source = _code[i].source;
      code[i].source_point = _code[i].source_point;
      code[i].weight = _code[i].weight;
      code[i].size = (int)_code[i].factor.size();
      code[i].factor = &factors[nfactors];
      memcpy(code[i].factor, &_code[i].factor[0], sizeof(cgpt_stencil_matrix_vector_factor_t) * code[i].size);
      nfactors += code[i].size;
    }
  }

  virtual ~cgpt_stencil_matrix_vector() {
  }
 
  virtual void execute(PVector<Lattice<M>> &matrix_fields, PVector<Lattice<V>> &vector_fields) {

    VECTOR_VIEW_OPEN(matrix_fields,fields_m_v,AcceleratorRead);
    VECTOR_VIEW_OPEN(vector_fields,fields_v_v,AcceleratorWrite);

    int n_code = code.size();
    const cgpt_stencil_matrix_vector_code_offload_t* p_code = &code[0];

    typedef decltype(coalescedRead(fields_v_v[0][0])) obj_v_t;
    typedef decltype(coalescedRead(fields_m_v[0][0])) obj_m_t;

    int nd = matrix_fields[0].Grid()->Nd();

    int _npb = n_code_parallel_blocks;
    int _npbs = n_code_parallel_block_size;
    
    auto sview = stencil.View();
    
    accelerator_for(ss_block,matrix_fields[0].Grid()->oSites() * _npb,M::Nsimd(),{
					    
        auto ss = ss_block / _npb;
	auto oblock = ss_block % _npb;

	for (int iblock=0;iblock<_npbs;iblock++) {

	  int i = oblock * _npbs + iblock;
	  obj_v_t t;

	  fetch(t, p_code[i].source_point, ss, fields_v_v[p_code[i].source], 0);

	  for (int j=p_code[i].size-1;j>=0;j--) {
	    obj_m_t f;
	    const auto _f = &p_code[i].factor[j];
	    fetch(f, _f->point, ss, fields_m_v[_f->index], _f->adj);
	    t = f * t;
	  }

	  obj_v_t r = p_code[i].weight * t;
	  if (p_code[i].accumulate != -1)
	    r += coalescedRead(fields_v_v[p_code[i].accumulate][ss]);
	  coalescedWrite(fields_v_v[p_code[i].target][ss], r);
	}
	
      });

    VECTOR_VIEW_CLOSE(fields_m_v);
    VECTOR_VIEW_CLOSE(fields_v_v);
  }

  virtual void execute(const std::vector<cgpt_Lattice_base*>& __matrix_fields, const std::vector<cgpt_Lattice_base*>& __vector_fields) {
    PVector<Lattice<M>> matrix_fields;
    PVector<Lattice<V>> vector_fields;
    cgpt_basis_fill(matrix_fields,__matrix_fields);
    cgpt_basis_fill(vector_fields,__vector_fields);
    execute(matrix_fields, vector_fields);
  }
};

static void cgpt_convert(PyObject* in, cgpt_stencil_matrix_vector_factor_t& out) {
  ASSERT(PyTuple_Check(in));
  ASSERT(PyTuple_Size(in) == 3);
  cgpt_convert(PyTuple_GetItem(in, 0), out.index);
  cgpt_convert(PyTuple_GetItem(in, 1), out.point);
  cgpt_convert(PyTuple_GetItem(in, 2), out.adj);
}

static void cgpt_convert(PyObject* in, cgpt_stencil_matrix_vector_code_t& out) {
  ASSERT(PyDict_Check(in));

  out.target = get_int(in, "target");
  out.accumulate = get_int(in, "accumulate");
  out.weight = get_complex(in, "weight");
  out.source = get_int(in, "source");
  out.source_point = get_int(in, "source_point");

  cgpt_convert(get_key(in, "factor"), out.factor);
}

template<typename V>
cgpt_stencil_matrix_vector_base*
cgpt_stencil_matrix_vector_create(cgpt_Lattice_base* __matrix, GridBase* grid, PyObject* _shifts, PyObject* _code,
				  long code_parallel_block_size, long local) {

  std::vector<Coordinate> shifts;
  cgpt_convert(_shifts,shifts);

  std::vector<cgpt_stencil_matrix_vector_code_t> code;
  cgpt_convert(_code,code);

  // for now only local is implemented
  ASSERT(local);

  // test __matrix type against matrix in spin space,
  // color space spin+color space, and singlet space
  if (is_compatible<typename matrixFromTypeAtLevel<V,2>::type>(__matrix)) {
    return new cgpt_stencil_matrix_vector<typename matrixFromTypeAtLevel<V,2>::type,V>(grid,shifts,code,code_parallel_block_size);
  } else if (is_compatible<typename matrixFromTypeAtLevel<V,1>::type>(__matrix)) {
    return new cgpt_stencil_matrix_vector<typename matrixFromTypeAtLevel<V,1>::type,V>(grid,shifts,code,code_parallel_block_size);
  } else if (is_compatible<typename matrixFromTypeAtLevel<V,0>::type>(__matrix)) {
    return new cgpt_stencil_matrix_vector<typename matrixFromTypeAtLevel<V,0>::type,V>(grid,shifts,code,code_parallel_block_size);
  } else if (is_compatible<typename matrixFromTypeAtLevel2<V,1,2>::type>(__matrix)) {
    return new cgpt_stencil_matrix_vector<typename matrixFromTypeAtLevel2<V,1,2>::type,V>(grid,shifts,code,code_parallel_block_size);
  } else {
    ERR("Unknown matrix type for matrix_vector stencil with vector type %s",typeid(V).name());
  }
}
