/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#define typeis(a,at) ( a->type() == typeid(at<vtype>).name() )
#define castas(a,at) ( ((cgpt_Lattice<at<vtype> >*)a)->l )
#define typeOpen(a,at) if (typeis(a,at)) { auto& l ## a = castas(a,at);
#define typeClose() }

// factor unary
#define BIT_TRANS 1
#define BIT_CONJ 2
#define NUM_FACTOR_UNARY 4

// term unary
#define BIT_SPINTRACE 1
#define BIT_COLORTRACE 2

template<typename T> cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T>& la,int unary_b, cgpt_Lattice_base* b, int unary_expr);
template<typename T> cgpt_Lattice_base* cgpt_compatible_linear_combination(Lattice<T>& _compatible,cgpt_Lattice_base* dst,bool ac, std::vector<cgpt_lattice_term>& f, int unary_factor, int unary_expr);

#include "expression/unary.h"
