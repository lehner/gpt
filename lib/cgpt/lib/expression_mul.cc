/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include "lib.h"

#include "expression/matmul.h"
#include "expression/mul.h"

#define PER_TENSOR_TYPE(T)						\
  INSTANTIATE(T,vComplexF)						\
  INSTANTIATE(T,vComplexD)						\

#define INSTANTIATE(T,vtype)						\
  template cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T<vtype>>& la,int unary_b, cgpt_Lattice_base* b, int unary_expr); \
  template cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T<vtype>>& la, PyArrayObject* b, int unary_expr, bool reverse);

#include "tensors.h"

#undef PER_TENSOR_TYPE
