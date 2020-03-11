/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#define EF(i) ((Coeff_t)f[i].get_coef()) * compatible<T>(f[i].get_lat())->l

// TODO: do unary f[i].unary here!! need to setup similar to mul, maybe can go to 4 or 5 terms;
// add a fallback option for more terms!

template<typename T>
cgpt_Lattice_base* cgpt_compatible_linear_combination(Lattice<T>& _compatible,cgpt_Lattice_base* dst,bool ac, std::vector<cgpt_lattice_term>& f, int unary_expr) {
  typedef typename Lattice<T>::scalar_type Coeff_t;
  int n = (int)f.size();
  if (n == 1) {
    return lattice_unary(dst,ac, EF(0), unary_expr );
  } else if (n == 2) {
    return lattice_unary(dst,ac, EF(0) + EF(1), unary_expr );
  } else if (n == 3) {
    return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2), unary_expr );
  } else if (n == 4) {
    return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2) + EF(3), unary_expr );
  } else if (n == 5) {
    return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2) + EF(3) + EF(4), unary_expr );
  } else if (n == 6) {
    return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2) + EF(3) + EF(4) + EF(5), unary_expr );
  } else if (n == 7) {
    return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2) + EF(3) + EF(4) + EF(5) + EF(6), unary_expr );
  } else if (n == 8) {
    return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2) + EF(3) + EF(4) + EF(5) + EF(6) + EF(7), unary_expr );
  } else if (n == 9) {
    return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2) + EF(3) + EF(4) + EF(5) + EF(6) + EF(7) + EF(8), unary_expr );
  } else {
    ERR("Need to hard-code linear combination n > 9");
  }
}

#undef EF

