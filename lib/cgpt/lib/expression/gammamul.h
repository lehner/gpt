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

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSinglet<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) {
  ERR("Not implemented");
}

#define SPIN(Ns)
#define SPIN_COLOR(Ns,Nc)

#define COLOR(Nc)							\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iVColor ## Nc<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) { \
    ERR("Not implemented");						\
  }									\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iMColor ## Nc<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) { \
    ERR("Not implemented");						\
  }									\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iVSpin4Color ## Nc<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) { \
    if (rev) {								\
      return lattice_unary_rmul(dst, ac, unary_a, la, Gamma(gamma), unary_expr); \
    }									\
    ERR("Not implemented");						\
  }									\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iMSpin4Color ## Nc<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) { \
    if (rev) {								\
      return lattice_unary_rmul(dst, ac, unary_a, la, Gamma(gamma), unary_expr); \
    } else {								\
      return lattice_unary_mul(dst, ac, unary_a, la, Gamma(gamma), unary_expr);	\
    }									\
  }									\

template<typename vtype> // spin4 vector
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iVSpin4<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) {
  if (rev) {
    return lattice_unary_rmul(dst, ac, unary_a, la, Gamma(gamma), unary_expr);
  }
  ERR("Not implemented");
}

template<typename vtype> // spin4 matrix
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iMSpin4<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) {
  if (rev) {
    return lattice_unary_rmul(dst, ac, unary_a, la, Gamma(gamma), unary_expr);
  } else {
    return lattice_unary_mul(dst, ac, unary_a, la, Gamma(gamma), unary_expr);
  }
  ERR("Not implemented");
}

template<typename vtype, int Ns> // general spin matrix
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iScalar<iMatrix<iScalar<vtype>, Ns> > >& la, Gamma::Algebra gamma, int unary_expr, bool rev) {
  ERR("Not implemented");
}

#include "../spin_color.h"
#undef SPIN
#undef COLOR
#undef SPIN_COLOR

#define BASIS_SIZE(n)							\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iVSinglet ## n<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) { \
    ERR("Not implemented");						\
  }									\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iMSinglet ## n<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) { \
    ERR("Not implemented");						\
  }
#include "../basis_size.h"
#undef BASIS_SIZE
