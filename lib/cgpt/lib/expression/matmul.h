/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#define typeOpen(a,at) { at< typename vtype::scalar_type > ab; std::vector<long> dim; cgpt_numpy_data_layout(ab,dim); if (cgpt_numpy_import(ab,a,dim)) {
#define typeClose() }}

template<typename A, typename B>
  cgpt_Lattice_base* lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, const A& la, const B& ab,int unary_expr) {
  if (unary_a == 0) {
    return lattice_unary(dst,ac, la*ab,unary_expr);
  } else if (unary_a == BIT_TRANS) {
    return lattice_unary(dst,ac, transpose(la)*ab,unary_expr);
  } else if (unary_a == BIT_CONJ) {
    return lattice_unary(dst,ac, conjugate(la)*ab,unary_expr);
  } else if (unary_a == BIT_CONJ|BIT_TRANS) {
    return lattice_unary(dst,ac, adj(la)*ab,unary_expr);
  }
  ERR("Not implemented");
}

template<typename A, typename B>
  cgpt_Lattice_base* lattice_rmatmul(cgpt_Lattice_base* dst, bool ac, int unary_a, const A& la, const B& ab,int unary_expr) {
  if (unary_a == 0) {
    return lattice_unary(dst,ac, ab*la, unary_expr);
  } else if (unary_a == BIT_TRANS) {
    return lattice_unary(dst,ac, ab*transpose(la), unary_expr);
  } else if (unary_a == BIT_CONJ) {
    return lattice_unary(dst,ac, ab*conjugate(la), unary_expr);
  } else if (unary_a == BIT_CONJ|BIT_TRANS) {
    return lattice_unary(dst,ac, ab*adj(la), unary_expr);
  }
  ERR("Not implemented");
}

#define _COMPATIBLE_RL_(t) typeOpen(b,t) { if (rev) { return lattice_rmatmul(dst,ac, unary_a,la,ab,unary_expr); } else { return lattice_matmul(dst,ac,unary_a,la,ab,unary_expr); } } typeClose();
#define _COMPATIBLE_R_(t) typeOpen(b,t) { if (rev) { return lattice_rmatmul(dst,ac, unary_a,la,ab,unary_expr); } else { ERR("Not supported"); } } typeClose();
#define _COMPATIBLE_L_(t) typeOpen(b,t) { if (rev) { ERR("Not supported"); } else { return lattice_matmul(dst,ac, unary_a,la,ab,unary_expr); } } typeClose();

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSinglet<vtype> >& la, PyArrayObject* b, int unary_expr, bool rev) {
  _COMPATIBLE_RL_(iColourVector);
  _COMPATIBLE_RL_(iColourMatrix);
  _COMPATIBLE_RL_(iSpinColourVector);
  _COMPATIBLE_RL_(iSpinColourMatrix);
  ERR("Not implemented");
}

#define _INNER_OUTER_PRODUCT_(t) ERR("Not implemented"); return 0;
//if (unary_a == BIT_TRANS|BIT_CONJ) { typeOpen(b,t) {			\
//    if (rev) { return lattice_unary_lat(dst, ac, outerProduct(ab,la), unary_expr ); } else \
//      { return lattice_unary_lat(dst, ac, localInnerProduct(la,ab), unary_expr ); } } typeClose(); }

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iColourVector<vtype> >& la, PyArrayObject* b, int unary_expr, bool rev) {
  _INNER_OUTER_PRODUCT_(iColourVector);
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iColourMatrix<vtype> >& la, PyArrayObject* b, int unary_expr, bool rev) {
  _COMPATIBLE_RL_(iColourMatrix);
  _COMPATIBLE_RL_(iSpinColourMatrix);
  _COMPATIBLE_L_(iColourVector);
  _COMPATIBLE_L_(iSpinColourVector);
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSpinColourVector<vtype> >& la, PyArrayObject* b, int unary_expr, bool rev) {
  _INNER_OUTER_PRODUCT_(iSpinColourVector);
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSpinColourMatrix<vtype> >& la, PyArrayObject* b, int unary_expr, bool rev) {
  _COMPATIBLE_RL_(iColourMatrix);
  _COMPATIBLE_RL_(iSpinColourMatrix);
  _COMPATIBLE_L_(iSpinColourVector);
  ERR("Not implemented");
}

#undef typeClose
#undef typeOpen
#undef _OUTER_PRODUCT_
#undef _INNER_PRODUCT_
#undef _COMPATIBLE_

