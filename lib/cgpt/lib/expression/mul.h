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
#define typeis(a,at) ( a->type() == typeid(at<vtype>).name() )
#define castas(a,at) ( ((cgpt_Lattice<at<vtype> >*)a)->l )
#define typeOpen(a,at) if (typeis(a,at)) { auto& l ## a = castas(a,at);
#define typeClose() }

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
  } else if (unary_b == (BIT_CONJ|BIT_TRANS)) {
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
  } else if (unary_a == (BIT_CONJ|BIT_TRANS)) {
    return lattice_mul(dst,ac, adj(la),unary_b,lb,unary_expr);
  }
  ERR("Not implemented");
}

#define _COMPATIBLE_(t) typeOpen(b,t) { return lattice_mul(dst,ac, unary_a,la,unary_b,lb,unary_expr); } typeClose();

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSinglet<vtype> >& la,int unary_b, cgpt_Lattice_base* b, int unary_expr) {
  //print(adj(Gamma::Algebra::Gamma5));
  _COMPATIBLE_(iSinglet);
  _COMPATIBLE_(iColourVector);
  _COMPATIBLE_(iColourMatrix);
  _COMPATIBLE_(iSpinColourVector);
  _COMPATIBLE_(iSpinColourMatrix);
  //#define BASIS_SIZE(n) _COMPATIBLE_(iComplexV ## n);
  //#include "../basis_size.h"
  //#undef BASIS_SIZE
  ERR("Not implemented");

#if 0
    GridCartesian* grid;
    Lattice< iColourVector<vComplexF> > a(grid),b(grid);
    Lattice< iColourMatrix<vComplexF> > m(grid);
    print( adj(a)*m );
#endif
}

#define _OUTER_PRODUCT_(t) if (unary_a == 0 && unary_b == (BIT_TRANS|BIT_CONJ)) { typeOpen(b,t) { return lattice_unary_lat(dst,ac, outerProduct(la,lb), unary_expr); } typeClose(); }
#define _INNER_PRODUCT_(t) if (unary_a == (BIT_TRANS|BIT_CONJ) && unary_b == 0) { typeOpen(b,t) { return lattice_unary_lat(dst, ac, localInnerProduct(la,lb), unary_expr ); } typeClose(); }

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iColourVector<vtype> >& la,int unary_b, cgpt_Lattice_base* b, int unary_expr) {
  _COMPATIBLE_(iSinglet);
  _OUTER_PRODUCT_(iColourVector);
  _INNER_PRODUCT_(iColourVector);
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iColourMatrix<vtype> >& la,int unary_b, cgpt_Lattice_base* b, int unary_expr) {
  _COMPATIBLE_(iSinglet);
  _COMPATIBLE_(iColourVector);
  _COMPATIBLE_(iColourMatrix);
  _COMPATIBLE_(iSpinColourVector);
  _COMPATIBLE_(iSpinColourMatrix);
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSpinColourVector<vtype> >& la,int unary_b, cgpt_Lattice_base* b, int unary_expr) {
  _COMPATIBLE_(iSinglet);
  _OUTER_PRODUCT_(iSpinColourVector);
  _INNER_PRODUCT_(iSpinColourVector);
  ERR("Not implemented");
}

// TODO: explicitly instantiate exact templates
template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSpinColourMatrix<vtype> >& la,int unary_b, cgpt_Lattice_base* b, int unary_expr) {
  _COMPATIBLE_(iSinglet);
  _COMPATIBLE_(iColourMatrix);
  _COMPATIBLE_(iSpinColourVector);
  _COMPATIBLE_(iSpinColourMatrix);
  ERR("Not implemented");
}

#define BASIS_SIZE(n)							\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iComplexV ## n<vtype> >& la,int unary_b, cgpt_Lattice_base* b, int unary_expr) { \
    ERR("Not implemented");						\
  }
#include "../basis_size.h"
#undef BASIS_SIZE

#undef typeClose
#undef typeOpen
#undef castas
#undef typeis
#undef _OUTER_PRODUCT_
#undef _INNER_PRODUCT_
#undef _COMPATIBLE_

