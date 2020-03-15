/*
  CGPT

  Authors: Christoph Lehner 2020
*/

template<typename A>
cgpt_Lattice_base* lattice_expr(cgpt_Lattice_base* dst, bool ac, const A& expr) {
  GridBase* grid;
  GridFromExpression(grid,expr);
  typedef decltype(eval(0,expr)) const_vobj;
  typedef typename std::remove_const<const_vobj>::type vobj;

  if (dst) {
    auto& l = compatible<vobj>(dst)->l;
    if (ac) {
      l = expr + l;
    } else {
      l = expr;
    }
    return dst;
  } else {
    ASSERT(!ac);
    cgpt_Lattice<vobj>* c = new cgpt_Lattice<vobj>((GridCartesian*)grid);
    c->l = expr;
    return (cgpt_Lattice_base*)c;
  }

}

template<typename A>
cgpt_Lattice_base* lattice_unary(cgpt_Lattice_base* dst, bool ac, const A& la,int unary_expr) {
  if (unary_expr == 0) {
    return lattice_expr(dst, ac, la);
  } else if (unary_expr == BIT_SPINTRACE|BIT_COLORTRACE) {
    return lattice_expr(dst, ac, trace(la));
  }
  ERR("Not implemented");
}

template<typename A> 
cgpt_Lattice_base* lattice_lat(cgpt_Lattice_base* dst, bool ac, const A& lat) {
  typedef typename A::vector_object const_vobj;
  typedef typename std::remove_const<const_vobj>::type vobj;
  if (dst) {
    auto& l = compatible<vobj>(dst)->l;
    if (ac) {
      l += lat;
    } else {
      l = lat;
    }
    return dst;
  } else {
    ASSERT(!ac);
    cgpt_Lattice<vobj>* c = new cgpt_Lattice<vobj>((GridCartesian*)lat.Grid());
    c->l = lat;
    return (cgpt_Lattice_base*)c;
  }
}

template<typename A>
cgpt_Lattice_base* lattice_unary_lat(cgpt_Lattice_base* dst, bool ac, const A& la,int unary_expr) {
  if (unary_expr == 0) {
    return lattice_lat(dst, ac, la);
  } else if (unary_expr == BIT_SPINTRACE|BIT_COLORTRACE) {
    return lattice_expr(dst, ac, trace(la));
  }
  ERR("Not implemented");
}

template<typename A, typename B>
  cgpt_Lattice_base* lattice_unary_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, const A& la, const B& ab,int unary_expr) {
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
  cgpt_Lattice_base* lattice_unary_rmul(cgpt_Lattice_base* dst, bool ac, int unary_a, const A& la, const B& ab,int unary_expr) {
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
