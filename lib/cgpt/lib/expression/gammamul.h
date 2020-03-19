/*
  CGPT

  Authors: Christoph Lehner 2020
*/

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSinglet<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) {
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iColourVector<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) {
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iColourMatrix<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) {
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSpinColourVector<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) {
  if (rev) {
    return lattice_unary_rmul(dst, ac, unary_a, la, Gamma(gamma), unary_expr);
  }
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSpinColourMatrix<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) {
  if (rev) {
    return lattice_unary_rmul(dst, ac, unary_a, la, Gamma(gamma), unary_expr);
  } else {
    return lattice_unary_mul(dst, ac, unary_a, la, Gamma(gamma), unary_expr);
  }
}
