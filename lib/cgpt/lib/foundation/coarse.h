/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2020  Daniel Richtmann (daniel.richtmann@ur.de, https://github.com/lehner/gpt)

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
    51 Franklin Street, Fifth Floor, Boston, MA 021basis_virtual_size-1n_virtual_red01 USA.
*/

template<class CComplex,int basis_virtual_size>
inline void invertCoarseLink(      PVector<Lattice<iMatrix<CComplex, basis_virtual_size>>>&   link_inv,
                             const PVector<Lattice<iMatrix<CComplex, basis_virtual_size>>>&   link,
                                   long                                                       n_virtual) {
  assert(link_inv.size() > 0 && link.size() > 0);

  assert(link_inv.size() % n_virtual == 0);
  assert(link_inv.size() == n_virtual);
  long link_inv_n = link_inv.size() / n_virtual;

  assert(link.size() % n_virtual == 0);
  assert(link_inv.size() == n_virtual);
  long link_n = link.size() / n_virtual;

  assert(link_inv_n == link_n);

  conformable(link_inv[0].Grid(), link[0].Grid());
  GridBase *grid = link[0].Grid();

  long lsites = grid->lSites();

  long n_virtual_red = long(sqrt(n_virtual));
  long nbasis_global = n_virtual_red * basis_virtual_size;

  typedef typename iMatrix<CComplex, basis_virtual_size>::scalar_object scalar_object;

  VECTOR_VIEW_OPEN(link_inv,link_inv_v,CpuWrite);
  VECTOR_VIEW_OPEN(link,link_v,CpuRead);

  thread_for(_idx, lsites, { // NOTE: Not on GPU because of Eigen & (peek/poke)LocalSite
    auto site = _idx;

    Eigen::MatrixXcd link_inv_eigen = Eigen::MatrixXcd::Zero(nbasis_global, nbasis_global);
    Eigen::MatrixXcd link_eigen = Eigen::MatrixXcd::Zero(nbasis_global, nbasis_global);

    scalar_object link_inv_tmp = Zero();
    scalar_object link_tmp = Zero();

    Coordinate lcoor;
    grid->LocalIndexToLocalCoor(site, lcoor);

    // convention for indices:
    // lex_outer = lexicographic index in array of v_objs
    // row_outer = row index    in array of v_objs (column-major ordering)
    // col_outer = column index in array of v_objs (column-major ordering)
    // row_inner = row index    inside a v_obj = grid matrix tensor (row-major ordering)
    // col_inner = column index inside a v_obj = grid matrix tensor (row-major ordering)
    // row_global = row index    of combination of v_objs viewed as 1 single big matrix tensor (row-major ordering)
    // col_global = column index of combination of v_objs viewed as 1 single big matrix tensor (row-major ordering)

    for (long lex_outer=0; lex_outer<n_virtual; lex_outer++) {
      peekLocalSite(link_tmp, link_v[lex_outer], lcoor);
      long row_outer = lex_outer % n_virtual_red;
      long col_outer = lex_outer / n_virtual_red;
      for (long row_inner=0; row_inner<basis_virtual_size; row_inner++) {
        for (long col_inner=0; col_inner<basis_virtual_size; col_inner++) {
          long row_global = row_outer * basis_virtual_size + row_inner;
          long col_global = col_outer * basis_virtual_size + col_inner;
          link_eigen(row_global, col_global) = static_cast<ComplexD>(TensorRemove(link_tmp(row_inner, col_inner)));
        }
      }
    }

    link_inv_eigen = link_eigen.inverse();

    for (long lex_outer=0; lex_outer<n_virtual; lex_outer++) {
      long row_outer = lex_outer % n_virtual_red;
      long col_outer = lex_outer / n_virtual_red;
      for (long row_inner=0; row_inner<basis_virtual_size; row_inner++) {
        for (long col_inner=0; col_inner<basis_virtual_size; col_inner++) {
          long row_global = row_outer * basis_virtual_size + row_inner;
          long col_global = col_outer * basis_virtual_size + col_inner;
          link_inv_tmp(row_inner, col_inner) = link_inv_eigen(row_global, col_global);
        }
      }
      pokeLocalSite(link_inv_tmp, link_inv_v[lex_outer], lcoor);
    }
  });

  VECTOR_VIEW_CLOSE(link_inv_v);
  VECTOR_VIEW_CLOSE(link_v);
}
