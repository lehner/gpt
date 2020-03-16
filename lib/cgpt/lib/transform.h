/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#define PER_TENSOR_TYPE(T)						\
  template<typename vtype>						\
  void cgpt_lattice_convert_from(Lattice< T<vtype> >& dst,cgpt_Lattice_base* src) { \
    if (src->type() == typeid(T<vComplexD>).name()) {			\
      precisionChange(dst, ((cgpt_Lattice<T<vComplexD>>*)src)->l );	\
    } else if (src->type() == typeid(T<vComplexF>).name()) {		\
      precisionChange(dst, ((cgpt_Lattice<T<vComplexF>>*)src)->l );	\
    } else {								\
      ERR("Only support conversion between single, double");		\
    }									\
  }
#include "tensors.h"
#undef PER_TENSOR_TYPE

template<typename T>
PyObject* cgpt_lattice_slice(Lattice<T>& l,int dim) {
  typedef typename Lattice<T>::vector_object vobj;
  typedef typename vobj::scalar_object sobj;

  std::vector<sobj> c;
  sliceSum(l,c,dim);

  PyObject* ret=PyList_New(c.size());
  for (size_t i=0; i<c.size(); i++) {
    PyList_SET_ITEM(ret,i,cgpt_numpy_export(c[i]));
  }
  return ret;
}
