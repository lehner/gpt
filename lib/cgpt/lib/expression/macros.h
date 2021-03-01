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

#define _MM_COMPATIBLE_RL_(t) _MM_typeOpen(b,t) { if (rev) { return lattice_unary_rmul(dst,ac, unary_a,la,ab,unary_expr,coef); } \
    else { return lattice_unary_mul(dst,ac,unary_a,la,ab,unary_expr,coef); } } _MM_typeClose();
#define _MM_COMPATIBLE_R_(t) _MM_typeOpen(b,t) { if (rev) { return lattice_unary_rmul(dst,ac, unary_a,la,ab,unary_expr,coef); } else { ERR("Not supported"); } } _MM_typeClose();
#define _MM_COMPATIBLE_L_(t) _MM_typeOpen(b,t) { if (rev) { ERR("Not supported"); } else { return lattice_unary_mul(dst,ac, unary_a,la,ab,unary_expr,coef); } } _MM_typeClose();

#define _MM_INNER_OUTER_PRODUCT_(t) ERR("Not implemented"); return 0;

#define typeis(a,at) ( a->type() == typeid(at<vtype>).name() )
#define castas(a,at) ( ((cgpt_Lattice<at<vtype> >*)a)->l )
#define typeOpen(a,at) if (typeis(a,at)) { auto& l ## a = castas(a,at);
#define typeClose() }

#define _COMPATIBLE_(t) typeOpen(b,t) { return lattice_mul(dst,ac, unary_a,la,unary_b,lb,unary_expr,coef); } typeClose();
#define _COMPATIBLE_MSR_(t) typeOpen(b,t) { return lattice_mul(dst,ac, unary_a,la,unary_b,lb,unary_expr,coef); } typeClose();

#define _OUTER_PRODUCT_(t) if (unary_a == 0 && unary_b == (BIT_TRANS|BIT_CONJ)) { typeOpen(b,t) { return lattice_unary_lat(dst,ac, outerProduct(la,lb), unary_expr,coef); } typeClose(); }
#define _INNER_PRODUCT_(t) if (unary_a == (BIT_TRANS|BIT_CONJ) && unary_b == 0) { typeOpen(b,t) { return lattice_unary_lat(dst, ac, localInnerProduct(la,lb), unary_expr,coef ); } typeClose(); }



#ifndef GRID_SIMT
#define DEF_z() typename result_type::vector_type v; zeroit(v);
#define DEF_o(O) O v; zeroit(v);
#else
#define DEF_z() typename result_type::scalar_type v; zeroit(v);
#define DEF_o(O) typename O::scalar_object v; zeroit(v);
#endif



#if defined(A64FX) || defined(A64FXFIXEDSIZE)
#define PREFETCH(a) {							\
    uint64_t base;							\
    base = (uint64_t)&a;						\
    for (int i=0;i<sizeof(a)/64;i++) {					\
      svprfd(svptrue_b64(), (int64_t*)(base +   i * 64), SV_PLDL2STRM);	\
    }									\
  }
#else
#define PREFETCH(a)
#endif
