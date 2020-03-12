/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include "lib.h"

#include "expression/mul.h"
#include "expression/linear_combination.h"

#define INSTANTIATE_OBJ(T)							\
  template cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T>& la,int unary_b, cgpt_Lattice_base* b, int unary_expr); \
  template cgpt_Lattice_base* cgpt_compatible_linear_combination(Lattice<T>& _compatible,cgpt_Lattice_base* dst,bool ac, std::vector<cgpt_lattice_term>& f, int unary_factor, int unary_expr);

#define INSTANTIATE_PREC(vtype)			\
  INSTANTIATE_OBJ(iSinglet<vtype>)		\
  INSTANTIATE_OBJ(iColourMatrix<vtype>)		\
  INSTANTIATE_OBJ(iColourVector<vtype>)		\
  INSTANTIATE_OBJ(iSpinColourMatrix<vtype>)	\
  INSTANTIATE_OBJ(iSpinColourVector<vtype>) 

INSTANTIATE_PREC(vComplexF);
INSTANTIATE_PREC(vComplexD);
