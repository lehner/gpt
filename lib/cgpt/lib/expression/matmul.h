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

#define _MM_typeOpen(a,at) { at< typename vtype::scalar_type > ab; if (bot == get_otype(ab)) { cgpt_numpy_import(ab,(PyObject*)a);
#define _MM_typeClose() }}

#define _MM_COMPATIBLE_RL_(t) _MM_typeOpen(b,t) { if (rev) { return lattice_unary_rmul(dst,ac, unary_a,la,ab,unary_expr); } else { return lattice_unary_mul(dst,ac,unary_a,la,ab,unary_expr); } } _MM_typeClose();
#define _MM_COMPATIBLE_R_(t) _MM_typeOpen(b,t) { if (rev) { return lattice_unary_rmul(dst,ac, unary_a,la,ab,unary_expr); } else { ERR("Not supported"); } } _MM_typeClose();
#define _MM_COMPATIBLE_L_(t) _MM_typeOpen(b,t) { if (rev) { ERR("Not supported"); } else { return lattice_unary_mul(dst,ac, unary_a,la,ab,unary_expr); } } _MM_typeClose();

#define _MM_INNER_OUTER_PRODUCT_(t) ERR("Not implemented"); return 0;

//if (unary_a == (BIT_TRANS|BIT_CONJ)) { typeOpen(b,t) {			\
//    if (rev) { return lattice_unary_lat(dst, ac, outerProduct(ab,la), unary_expr ); } else \
//      { return lattice_unary_lat(dst, ac, localInnerProduct(la,ab), unary_expr ); } } typeClose(); }

//if (!rev && unary_b == (BIT_TRANS|BIT_CONJ) && unary_a == 0) {
//  typeOpen(b,iSpinColourVector) { return lattice_unary_lat(dst,ac, outerProduct(la,ab), unary_expr); } typeClose();
//}

