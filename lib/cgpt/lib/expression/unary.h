/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

template<typename A>
cgpt_Lattice_base* lattice_expr(cgpt_Lattice_base* dst, bool ac, const A& expr) {
  GridBase* grid = 0;
  GridFromExpression(grid,expr);
  typedef decltype(vecEval(0,expr)) const_vobj;
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
  } else if (unary_expr == (BIT_SPINTRACE|BIT_COLORTRACE)) {
    return lattice_expr(dst, ac, ToSinglet(trace(la)));
  } else if (unary_expr == BIT_SPINTRACE) {
    return lattice_lat(dst, ac, TraceIndex<SpinIndex>(closure(ToSinglet(la))),1.0);
  } else if (unary_expr == BIT_COLORTRACE) {
    return lattice_lat(dst, ac, TraceIndex<ColourIndex>(closure(ToSinglet(la))),1.0);
  }
  ERR("Not implemented");
}

template<typename A> 
cgpt_Lattice_base* lattice_lat(cgpt_Lattice_base* dst, bool ac, const A& lat, ComplexD coef) {
  typedef typename A::vector_object const_vobj;
  typedef typename std::remove_const<const_vobj>::type vobj;
  if (dst) {
    auto& l = compatible<vobj>(dst)->l;
    if (ac) {
      l += coef * lat;
    } else {
      l = coef * lat;
    }
    return dst;
  } else {
    ASSERT(!ac);
    cgpt_Lattice<vobj>* c = new cgpt_Lattice<vobj>((GridCartesian*)lat.Grid());
    c->l = coef * lat;
    return (cgpt_Lattice_base*)c;
  }
}

template<typename A>
cgpt_Lattice_base* lattice_unary_lat(cgpt_Lattice_base* dst, bool ac, const A& la,int unary_expr,ComplexD coef) {
  if (unary_expr == 0) {
    return lattice_lat(dst, ac, la, coef);
  } else if (unary_expr == (BIT_SPINTRACE|BIT_COLORTRACE)) {
    return lattice_expr(dst, ac, coef*ToSinglet(trace(la)));
  } else if (unary_expr == BIT_SPINTRACE) {
    return lattice_lat(dst, ac, TraceIndex<SpinIndex>(closure(ToSinglet(la))),coef);
  } else if (unary_expr == BIT_COLORTRACE) {
    return lattice_lat(dst, ac, TraceIndex<ColourIndex>(closure(ToSinglet(la))),coef);
  }
  ERR("Not implemented");
}

