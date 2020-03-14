/*
  CGPT

  Authors: Christoph Lehner 2020
*/
// factor unary
#define BIT_TRANS 1
#define BIT_CONJ 2
#define NUM_FACTOR_UNARY 4

// term unary
#define BIT_SPINTRACE 1
#define BIT_COLORTRACE 2

// need to supplement this for current Grid
template<class vtype,int N> accelerator_inline iVector<vtype,N> transpose(const iVector<vtype,N>&r) { return r; }

// declaration
template<typename T> cgpt_Lattice_base* cgpt_compatible_linear_combination(Lattice<T>& _compatible,cgpt_Lattice_base* dst,bool ac, std::vector<cgpt_lattice_term>& f, int unary_factor, int unary_expr);

#define PER_TENSOR_TYPE(T) \
  template<typename vtype> cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T<vtype>>& la,int unary_b, cgpt_Lattice_base* b, int unary_expr); \
  template<typename vtype> cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T<vtype>>& la, PyArrayObject* b, int unary_expr, bool reverse);

#include "tensors.h"

#undef PER_TENSOR_TYPE

// unary
#include "expression/unary.h"
