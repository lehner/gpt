/*
  CGPT

  Authors: Christoph Lehner 2020
*/

static int algebra_map_max = 5;

static Gamma::Algebra algebra_map[] = {
  Gamma::Algebra::GammaX, // 0
  Gamma::Algebra::GammaY, // 1
  Gamma::Algebra::GammaZ, // 2
  Gamma::Algebra::GammaT, // 3
  Gamma::Algebra::Gamma5  // 4
};

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSinglet<vtype> >& la, int gamma, int unary_expr, bool rev) {
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iColourVector<vtype> >& la, int gamma, int unary_expr, bool rev) {
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iColourMatrix<vtype> >& la, int gamma, int unary_expr, bool rev) {
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSpinColourVector<vtype> >& la, int gamma, int unary_expr, bool rev) {
  if (rev) {
    ASSERT(gamma >= 0 && gamma < algebra_map_max);
    return lattice_unary_rmul(dst, ac, unary_a, la, Gamma(algebra_map[gamma]), unary_expr);
  }
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSpinColourMatrix<vtype> >& la, int gamma, int unary_expr, bool rev) {
  ASSERT(gamma >= 0 && gamma < algebra_map_max);
  if (rev) {
    return lattice_unary_rmul(dst, ac, unary_a, la, Gamma(algebra_map[gamma]), unary_expr);
  } else {
    return lattice_unary_mul(dst, ac, unary_a, la, Gamma(algebra_map[gamma]), unary_expr);
  }
}

