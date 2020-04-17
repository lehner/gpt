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

#define _LC_(EF)							\
  if (n == 1) {								\
    return lattice_unary(dst,ac, EF(0), unary_expr );			\
  } else if (n == 2) {							\
    return lattice_unary(dst,ac, EF(0) + EF(1), unary_expr );		\
  } else if (n == 3) {							\
    return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2), unary_expr );	\
  } else if (n == 4) {							\
    return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2) + EF(3), unary_expr ); \
  } else if (n == 5) {							\
    return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2) + EF(3) + EF(4), unary_expr ); \
  } else if (n == 6) {							\
    return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2) + EF(3) + EF(4) + EF(5), unary_expr ); \
  } else if (n == 7) {							\
    return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2) + EF(3) + EF(4) + EF(5) + EF(6), unary_expr ); \
  } else {								\
    ERR("Need to hard-code linear combination n > 7");			\
  }
  /*} else if (n == 8) {						\
    return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2) + EF(3) + EF(4) + EF(5) + EF(6) + EF(7), unary_expr ); \
  } else if (n == 9) {							\
  return lattice_unary(dst,ac, EF(0) + EF(1) + EF(2) + EF(3) + EF(4) + EF(5) + EF(6) + EF(7) + EF(8), unary_expr ); \*/

#define EF(i) ((Coeff_t)f[i].get_coef()) * compatible<T>(f[i].get_lat())->l
#define EF_transpose(i) ((Coeff_t)f[i].get_coef()) * transpose(compatible<T>(f[i].get_lat())->l)
#define EF_conj(i) ((Coeff_t)f[i].get_coef()) * conjugate(compatible<T>(f[i].get_lat())->l)
#define EF_adj(i) ((Coeff_t)f[i].get_coef()) * adj(compatible<T>(f[i].get_lat())->l)

template<typename T>
cgpt_Lattice_base* cgpt_compatible_linear_combination(Lattice<T>& _compatible,cgpt_Lattice_base* dst,bool ac, std::vector<cgpt_lattice_term>& f, int unary_factor, int unary_expr) {
  typedef typename Lattice<T>::scalar_type Coeff_t;
  int n = (int)f.size();
  if (unary_factor == 0) {
    _LC_(EF);
  } else if (unary_factor == BIT_TRANS) {
    _LC_(EF_transpose);
  } else if (unary_factor == BIT_CONJ) {
    _LC_(EF_conj);
  } else if (unary_factor == (BIT_TRANS|BIT_CONJ)) {
    _LC_(EF_adj);
  } else {
    ERR("Not implemented");
  }
}

#undef EF
#undef EF_transpose
#undef EF_conj
#undef EF_adj
#undef _LC_


