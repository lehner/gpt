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
#define typeOpen(a,at) { at< typename vtype::scalar_type > ab; if (bot == get_otype(ab)) { cgpt_numpy_import(ab,(PyObject*)a);
#define typeClose() }}

#define _COMPATIBLE_RL_(t) typeOpen(b,t) { if (rev) { return lattice_unary_rmul(dst,ac, unary_a,la,ab,unary_expr); } else { return lattice_unary_mul(dst,ac,unary_a,la,ab,unary_expr); } } typeClose();
#define _COMPATIBLE_R_(t) typeOpen(b,t) { if (rev) { return lattice_unary_rmul(dst,ac, unary_a,la,ab,unary_expr); } else { ERR("Not supported"); } } typeClose();
#define _COMPATIBLE_L_(t) typeOpen(b,t) { if (rev) { ERR("Not supported"); } else { return lattice_unary_mul(dst,ac, unary_a,la,ab,unary_expr); } } typeClose();

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSinglet<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) {
  if (unary_b == 0) {
#define COLOR(Nc)				\
    _COMPATIBLE_RL_(iVColor ## Nc);		\
    _COMPATIBLE_RL_(iMColor ## Nc);
#define SPIN(Ns)				\
    _COMPATIBLE_RL_(iVSpin ## Ns);		\
    _COMPATIBLE_RL_(iMSpin ## Ns);
#define SPIN_COLOR(Ns,Nc)				\
    _COMPATIBLE_RL_(iVSpin ## Ns ## Color ## Nc);	\
    _COMPATIBLE_RL_(iMSpin ## Ns ## Color ## Nc);
#include "../spin_color.h"
#undef COLOR
#undef SPIN
#undef SPIN_COLOR
  }
  ERR("Not implemented");
}

#define _INNER_OUTER_PRODUCT_(t) ERR("Not implemented"); return 0;
//if (unary_a == (BIT_TRANS|BIT_CONJ)) { typeOpen(b,t) {			\
//    if (rev) { return lattice_unary_lat(dst, ac, outerProduct(ab,la), unary_expr ); } else \
//      { return lattice_unary_lat(dst, ac, localInnerProduct(la,ab), unary_expr ); } } typeClose(); }

#define COLOR(Nc)							\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iVColor ## Nc<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) { \
    _INNER_OUTER_PRODUCT_(iVColor ## Nc);				\
    ERR("Not implemented");						\
  }									\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iMColor ## Nc<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) { \
    if (unary_b == 0) {							\
      _COMPATIBLE_RL_(iMColor ## Nc);					\
      _COMPATIBLE_L_(iVColor ## Nc);					\
      _COMPATIBLE_RL_(iMSpin4Color ## Nc);				\
      _COMPATIBLE_L_(iVSpin4Color ## Nc);				\
    }									\
    ERR("Not implemented");						\
  }

#define SPIN(Ns)							\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iVSpin ## Ns<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) { \
    _INNER_OUTER_PRODUCT_(iVColor ## Nc);				\
    ERR("Not implemented");						\
  }									\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iMSpin ## Ns<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) { \
    if (unary_b == 0) {							\
      _COMPATIBLE_RL_(iMSpin ## Ns);					\
      _COMPATIBLE_L_(iVSpin ## Ns);					\
      _COMPATIBLE_RL_(iMSpin ## Ns ## Color3);				\
      _COMPATIBLE_L_(iVSpin ## Ns ## Color3);				\
    }									\
    ERR("Not implemented");						\
  }

//if (!rev && unary_b == (BIT_TRANS|BIT_CONJ) && unary_a == 0) {
//  typeOpen(b,iSpinColourVector) { return lattice_unary_lat(dst,ac, outerProduct(la,ab), unary_expr); } typeClose();
//}
#define SPIN_COLOR(Ns,Nc)						\
  template<typename vtype>						\
    cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iVSpin ## Ns ## Color ## Nc<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) { \
    _INNER_OUTER_PRODUCT_(iVSpin ## Ns ## Color ## Nc);			\
    ERR("Not implemented");						\
  }									\
  template<typename vtype>						\
    cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iMSpin ## Ns ## Color ## Nc<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) { \
    if (unary_b == 0) {							\
      _COMPATIBLE_RL_(iMColor ## Nc);					\
      _COMPATIBLE_RL_(iMSpin ## Ns);					\
      _COMPATIBLE_RL_(iMSpin ## Ns ## Color ## Nc);			\
      _COMPATIBLE_L_(iVSpin ## Ns ## Color ## Nc);			\
    }									\
    ERR("Not implemented");						\
  }

#include "../spin_color.h"

#undef SPIN_COLOR
#undef SPIN
#undef COLOR

#define BASIS_SIZE(n)							\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iVSinglet ## n<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) { \
    ERR("Not implemented");						\
  }									\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iMSinglet ## n<vtype> >& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool rev) { \
    ERR("Not implemented");						\
  }
#include "../basis_size.h"
#undef BASIS_SIZE


#undef typeClose
#undef typeOpen
#undef _OUTER_PRODUCT_
#undef _INNER_PRODUCT_
#undef _COMPATIBLE_

