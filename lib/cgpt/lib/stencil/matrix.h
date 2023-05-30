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

#ifndef GRID_HAS_ACCELERATOR

// cpu fetch version
#define fetch(obj, point, site, view, do_adj) {				\
    auto _SE = sview.GetEntry(point,site);				\
    obj = coalescedRead(view[_SE->_offset]);				\
    auto tmp = obj;							\
    if (_SE->_permute)							\
      for (int d=0;d<nd;d++)						\
	if (_SE->_permute & (0x1 << d)) { permute(obj,tmp,d); tmp=obj;}	\
    if (do_adj)								\
      obj = adj(obj);							\
  }

#else

template<class vobj> accelerator_inline
typename vobj::scalar_object coalescedReadGeneralPermute(const vobj & __restrict__ vec,int permute,int nd,int lane=acceleratorSIMTlane(vobj::Nsimd()))
{
  int plane = lane;
  for (int d=0;d<nd;d++)
    plane = (permute & (0x1 << d)) ? plane ^ (vobj::Nsimd() >> (d + 1)) : plane;
  return extractLane(plane,vec);
}

// gpu fetch version
#define fetch(obj, point, site, view, do_adj) {				\
    auto _SE = sview.GetEntry(point,site);				\
    if (_SE->_permute) {						\
      obj = coalescedReadGeneralPermute(view[_SE->_offset], _SE->_permute,nd); \
    } else {								\
      obj = coalescedRead(view[_SE->_offset]);				\
    }									\
    if (do_adj)								\
      obj = adj(obj);							\
  }

#endif
  
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

#include <Grid/stencil/GeneralLocalStencil.h>

class cgpt_stencil_matrix_base {
 public:
  virtual ~cgpt_stencil_matrix_base() {};
  virtual void execute(const std::vector<cgpt_Lattice_base*>& fields) = 0;
};

template<typename T>
void cgpt_basis_fill(PVector<Lattice<T>>& basis, const std::vector<cgpt_Lattice_base*>& _basis);

template<typename T>
class cgpt_stencil_matrix : public cgpt_stencil_matrix_base {
 public:

  GeneralLocalStencil stencil;
  Vector<cgpt_stencil_matrix_code_offload_t> code;
  Vector<cgpt_stencil_matrix_factor_t> factors;
    
  cgpt_stencil_matrix(GridBase* grid,
		      const std::vector<Coordinate>& shifts,
		      const std::vector<cgpt_stencil_matrix_code_t>& _code) :
  stencil(grid,shifts), code(_code.size()) {
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
  }

  virtual ~cgpt_stencil_matrix() {
  }
 
  virtual void execute(PVector<Lattice<T>> &fields) {

    VECTOR_VIEW_OPEN(fields,fields_v,AcceleratorWrite);

    int n_code = code.size();
    cgpt_stencil_matrix_code_offload_t* p_code = &code[0];

    typedef decltype(coalescedRead(fields_v[0][0])) obj_t;

    int nd = fields[0].Grid()->Nd();

    auto sview = stencil.View();
    
    accelerator_for(ss,fields[0].Grid()->oSites(),T::Nsimd(),{

	for (int i=0;i<n_code;i++) {
	  obj_t t;

	  auto _f0 = &p_code[i].factor[0];
	  fetch(t, _f0->point, ss, fields_v[_f0->index], _f0->adj);

	  for (int j=1;j<p_code[i].size;j++) {
	    obj_t f;
	    auto _f = &p_code[i].factor[j];
	    fetch(f, _f->point, ss, fields_v[_f->index], _f->adj);
	    t = t * f;
	  }

	  obj_t r = p_code[i].weight * t;
	  if (p_code[i].accumulate != -1)
	    r += coalescedRead(fields_v[p_code[i].accumulate][ss]);
	  coalescedWrite(fields_v[p_code[i].target][ss], r);
	}
	
      });
    
    VECTOR_VIEW_CLOSE(fields_v);
  }

  virtual void execute(const std::vector<cgpt_Lattice_base*>& __fields) {
    PVector<Lattice<T>> fields;
    cgpt_basis_fill(fields,__fields);
    execute(fields);
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

// traits to identify types for which type(a*a) = type(a)
template<typename T>     struct isEndomorphism                : public std::true_type { static constexpr bool notvalue = false; };
template<typename T>     struct isEndomorphism<iScalar<T>>    : public isEndomorphism<T> { static constexpr bool notvalue = isEndomorphism<T>::notvalue; };
template<typename T, int N>     struct isEndomorphism<iMatrix<T,N>>    : public isEndomorphism<T> { static constexpr bool notvalue = isEndomorphism<T>::notvalue; };
template<typename T, int N>     struct isEndomorphism<iVector<T,N>>    : public std::false_type { static constexpr bool notvalue = true; };

// not implemented message
template<typename T>
NotEnableIf<isEndomorphism<T>,cgpt_stencil_matrix_base*>
cgpt_stencil_matrix_create(GridBase* grid, PyObject* _shifts, PyObject* _code) {
  ERR("cgpt_stencil_matrix not implemented for type %s",typeid(T).name());
}

// implemented for endomorphisms
template<typename T>
EnableIf<isEndomorphism<T>,cgpt_stencil_matrix_base*>
cgpt_stencil_matrix_create(GridBase* grid, PyObject* _shifts, PyObject* _code) {

  std::vector<Coordinate> shifts;
  cgpt_convert(_shifts,shifts);

  std::vector<cgpt_stencil_matrix_code_t> code;
  cgpt_convert(_code,code);
  
  return new cgpt_stencil_matrix<T>(grid,shifts,code);
}

#undef fetch
