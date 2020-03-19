/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include "lib.h"

#include "expression/gammamul.h"

#define PER_TENSOR_TYPE(T)						\
  INSTANTIATE(T,vComplexF)						\
  INSTANTIATE(T,vComplexD)						\

#define INSTANTIATE(T,vtype)						\
  template cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T<vtype>>& la, Gamma::Algebra gamma, int unary_expr, bool rev);

#include "tensors.h"

#undef PER_TENSOR_TYPE
