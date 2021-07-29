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
#include "lib.h"

EXPORT(invert_matrix,{

    PyObject * _matrix_inv, * _matrix;
    if (!PyArg_ParseTuple(args, "OO", &_matrix_inv, &_matrix)) {
      return NULL;
    }

    std::vector<cgpt_Lattice_base*> matrix_inv;
    long matrix_inv_n_virtual = cgpt_basis_fill(matrix_inv, _matrix_inv);

    std::vector<cgpt_Lattice_base*> matrix;
    long matrix_n_virtual = cgpt_basis_fill(matrix, _matrix);

    ASSERT(matrix_inv_n_virtual == matrix_n_virtual);

    ASSERT(matrix.size() > 0);

    matrix[0]->invert_matrix(matrix_inv, matrix, matrix_n_virtual);

    return PyLong_FromLong(0);
  });

EXPORT(determinant,{

    PyObject * _matrix;
    void* _det;
    if (!PyArg_ParseTuple(args, "lO", &_det, &_matrix)) {
      return NULL;
    }

    cgpt_Lattice_base* det = (cgpt_Lattice_base*)_det;
    std::vector<cgpt_Lattice_base*> matrix;
    long matrix_n_virtual = cgpt_basis_fill(matrix, _matrix);

    ASSERT(matrix.size() > 0);

    matrix[0]->determinant(det, matrix, matrix_n_virtual);

    return PyLong_FromLong(0);
  });
