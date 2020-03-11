/*
  CGPT

  Authors: Christoph Lehner 2020
*/

/*
template<typename A> inline cgpt_Lattice_base* lattice_ret(const A& lat) {
  typedef typename A::vector_object const_vobj;
  typedef typename std::remove_const<const_vobj>::type vobj;
  cgpt_Lattice<vobj>* c = new cgpt_Lattice<vobj>((GridCartesian*)lat.Grid());
  c->l = lat;
  return (cgpt_Lattice_base*)c;
}
*/

template<typename A, typename B>
  cgpt_Lattice_base* lattice_mul(cgpt_Lattice_base* dst, bool ac, const A& la, const B& lb,int unary_expr) {
  return lattice_unary(dst, ac, la*lb, unary_expr );
}

template<typename A, typename B>
  cgpt_Lattice_base* lattice_mul(cgpt_Lattice_base* dst, bool ac, const A& la, int unary_b, const B& lb,int unary_expr) {
  if (unary_b == 0) {
    return lattice_mul(dst,ac,la,lb,unary_expr);
  } else if (unary_b == BIT_TRANS) {
    return lattice_mul(dst,ac,la,transpose(lb),unary_expr);
  } else if (unary_b == BIT_CONJ) {
    return lattice_mul(dst,ac,la,conjugate(lb),unary_expr);
  } else if (unary_b == BIT_CONJ|BIT_TRANS) {
    return lattice_mul(dst,ac,la,adj(lb),unary_expr);
  }
  ERR("Not implemented");
}

template<typename A, typename B>
  cgpt_Lattice_base* lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, const A& la, int unary_b, const B& lb,int unary_expr) {
  if (unary_a == 0) {
    return lattice_mul(dst,ac, la,unary_b,lb,unary_expr);
  } else if (unary_a == BIT_TRANS) {
    return lattice_mul(dst,ac, transpose(la),unary_b,lb,unary_expr);
  } else if (unary_a == BIT_CONJ) {
    return lattice_mul(dst,ac, conjugate(la),unary_b,lb,unary_expr);
  } else if (unary_a == BIT_CONJ|BIT_TRANS) {
    return lattice_mul(dst,ac, adj(la),unary_b,lb,unary_expr);
  }
  ERR("Not implemented");
}

#define _COMPATIBLE_(t) typeOpen(b,t) { return lattice_mul(dst,ac, unary_a,la,unary_b,lb,unary_expr); } typeClose();

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSinglet<vtype> >& la,int unary_b, cgpt_Lattice_base* b, int unary_expr) {
  //print(adj(Gamma::Algebra::Gamma5));
  _COMPATIBLE_(iSinglet);
  _COMPATIBLE_(iColourMatrix);
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iColourMatrix<vtype> >& la,int unary_b, cgpt_Lattice_base* b, int unary_expr) {
  _COMPATIBLE_(iColourMatrix);  
  ERR("Not implemented");
}

#undef _COMPATIBLE_

