/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#define typeOpen(a,at) { at< typename vtype::scalar_type > ab; if (bot == get_otype(ab)) { cgpt_numpy_import(ab,(PyObject*)a);
#define typeClose() }}

#define _COMPATIBLE_RL_(t) typeOpen(b,t) { if (rev) { return lattice_unary_rmul(dst,ac, unary_a,la,ab,unary_expr); } else { return lattice_unary_mul(dst,ac,unary_a,la,ab,unary_expr); } } typeClose();
#define _COMPATIBLE_R_(t) typeOpen(b,t) { if (rev) { return lattice_unary_rmul(dst,ac, unary_a,la,ab,unary_expr); } else { ERR("Not supported"); } } typeClose();
#define _COMPATIBLE_L_(t) typeOpen(b,t) { if (rev) { ERR("Not supported"); } else { return lattice_unary_mul(dst,ac, unary_a,la,ab,unary_expr); } } typeClose();

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSinglet<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) {
  if (unary_b == 0) {
    _COMPATIBLE_RL_(iColourVector);
    _COMPATIBLE_RL_(iColourMatrix);
    _COMPATIBLE_RL_(iSpinColourVector);
    _COMPATIBLE_RL_(iSpinColourMatrix);
  }
  ERR("Not implemented");
}

#define _INNER_OUTER_PRODUCT_(t) ERR("Not implemented"); return 0;
//if (unary_a == BIT_TRANS|BIT_CONJ) { typeOpen(b,t) {			\
//    if (rev) { return lattice_unary_lat(dst, ac, outerProduct(ab,la), unary_expr ); } else \
//      { return lattice_unary_lat(dst, ac, localInnerProduct(la,ab), unary_expr ); } } typeClose(); }

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iColourVector<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) {
  _INNER_OUTER_PRODUCT_(iColourVector);
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iColourMatrix<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) {
  if (unary_b == 0) {
    _COMPATIBLE_RL_(iColourMatrix);
    _COMPATIBLE_RL_(iSpinColourMatrix);
    _COMPATIBLE_L_(iColourVector);
    _COMPATIBLE_L_(iSpinColourVector);
  }
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSpinColourVector<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) {
  _INNER_OUTER_PRODUCT_(iSpinColourVector);
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSpinColourMatrix<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) {
  if (unary_b == 0) {
    _COMPATIBLE_RL_(iColourMatrix);
    _COMPATIBLE_RL_(iSpinColourMatrix);
    _COMPATIBLE_L_(iSpinColourVector);
  }
  ERR("Not implemented");
}

#undef typeClose
#undef typeOpen
#undef _OUTER_PRODUCT_
#undef _INNER_PRODUCT_
#undef _COMPATIBLE_

