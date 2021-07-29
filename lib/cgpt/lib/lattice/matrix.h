/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2020  Daniel Richtmann (daniel.richtmann@ur.de)

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
template<typename T>
void cgpt_invert_matrix(Lattice<T>& l, std::vector<cgpt_Lattice_base*>& _matrix_inv, std::vector<cgpt_Lattice_base*>& _matrix, long n_virtual) {
  
  PVector<Lattice<T>> matrix_inv;
  PVector<Lattice<T>> matrix;
  cgpt_basis_fill(matrix_inv, _matrix_inv);
  cgpt_basis_fill(matrix, _matrix);

  ASSERT(matrix.size() == matrix_inv.size());
  ASSERT(matrix.size() > 0 && matrix.size() % n_virtual == 0);

  invertMatrix(matrix_inv, matrix, n_virtual);
}

template<typename T>
void cgpt_determinant(Lattice<T>& l, cgpt_Lattice_base* _det, std::vector<cgpt_Lattice_base*>& _matrix, long n_virtual) {

  typedef typename Lattice<T>::vector_type vCoeff_t;
  
  Lattice< iSinglet<vCoeff_t> > & det = compatible< iSinglet<vCoeff_t> >(_det)->l;
  
  PVector<Lattice<T>> matrix;
  cgpt_basis_fill(matrix, _matrix);

  ASSERT(matrix.size() > 0 && matrix.size() % n_virtual == 0);

  determinant(det, matrix, n_virtual);
}
