/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include "lib.h"

#include "expression/linear_combination.h"

#define PER_TENSOR_TYPE(T)						\
  INSTANTIATE(T,vComplexF)						\
  INSTANTIATE(T,vComplexD)						\

#define INSTANTIATE(T,vtype)						\
  template cgpt_Lattice_base* cgpt_compatible_linear_combination(Lattice<T<vtype>>& _compatible,cgpt_Lattice_base* dst,bool ac, std::vector<cgpt_lattice_term>& f, int unary_factor, int unary_expr);

#include "tensors.h"

#undef PER_TENSOR_TYPE
