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
#include "lib.h"

EXPORT(block_project,{

    PyObject* _basis;
    void* _coarse,* _fine;
    int idx;
    if (!PyArg_ParseTuple(args, "llOi", &_coarse,&_fine,&_basis,&idx)) {
      return NULL;
    }

    cgpt_Lattice_base* fine = (cgpt_Lattice_base*)_fine;
    cgpt_Lattice_base* coarse = (cgpt_Lattice_base*)_coarse;

    std::vector<cgpt_Lattice_base*> basis;
    cgpt_basis_fill(basis,_basis,idx);

    fine->block_project(coarse,basis);

    return PyLong_FromLong(0);
  });

EXPORT(block_promote,{

    PyObject* _basis;
    void* _coarse,* _fine;
    int idx;
    if (!PyArg_ParseTuple(args, "llOi", &_coarse,&_fine,&_basis,&idx)) {
      return NULL;
    }

    cgpt_Lattice_base* fine = (cgpt_Lattice_base*)_fine;
    cgpt_Lattice_base* coarse = (cgpt_Lattice_base*)_coarse;

    std::vector<cgpt_Lattice_base*> basis;
    cgpt_basis_fill(basis,_basis,idx);

    fine->block_promote(coarse,basis);

    return PyLong_FromLong(0);
  });

EXPORT(block_orthonormalize,{

    PyObject* _basis;
    void* _coarse;
    if (!PyArg_ParseTuple(args, "lO", &_coarse,&_basis)) {
      return NULL;
    }

    cgpt_Lattice_base* coarse = (cgpt_Lattice_base*)_coarse;

    std::vector<cgpt_Lattice_base*> basis;
    cgpt_basis_fill(basis,_basis,0); // TODO: generalize

    ASSERT(basis.size() > 0);
    basis[0]->block_orthonormalize(coarse,basis);

    return PyLong_FromLong(0);
  });

