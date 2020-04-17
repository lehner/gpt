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

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iColourVector<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) {
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iColourMatrix<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) {
  ERR("Not implemented");
}

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSpinColourVector<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) {
  if (rev) {
    return lattice_unary_rmul(dst, ac, unary_a, la, Gamma(gamma), unary_expr);
  }
  ERR("Not implemented");
}

#define BASIS_SIZE(n)							\
  template<typename vtype>						\
  cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iComplexV ## n<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) { \
    ERR("Not implemented");						\
  }
#include "../basis_size.h"
#undef BASIS_SIZE

template<typename vtype>
cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice< iSpinColourMatrix<vtype> >& la, Gamma::Algebra gamma, int unary_expr, bool rev) {
  if (rev) {
    return lattice_unary_rmul(dst, ac, unary_a, la, Gamma(gamma), unary_expr);
  } else {
    return lattice_unary_mul(dst, ac, unary_a, la, Gamma(gamma), unary_expr);
  }
}
