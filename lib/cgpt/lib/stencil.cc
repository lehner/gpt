/*
    GPT - Grid Python Toolkit
    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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

EXPORT(stencil_matrix_create,{

    void* _grid;
    void* _lattice;
    PyObject* _shifts, * _code;
    if (!PyArg_ParseTuple(args, "llOO", &_lattice, &_grid, &_shifts, &_code)) {
      return NULL;
    }
    
    GridBase* grid = (GridBase*)_grid;
    cgpt_Lattice_base* lattice = (cgpt_Lattice_base*)_lattice;

    return PyLong_FromVoidPtr(lattice->stencil_matrix(grid, _shifts, _code));
  });

EXPORT(stencil_matrix_execute,{

    void* _stencil;
    PyObject* _fields;
    if (!PyArg_ParseTuple(args, "lO", &_stencil, &_fields)) {
      return NULL;
    }
    
    cgpt_stencil_matrix_base* stencil = (cgpt_stencil_matrix_base*)_stencil;

    std::vector<cgpt_Lattice_base*> __fields;
    cgpt_basis_fill(__fields,_fields);

    stencil->execute(__fields);

    return PyLong_FromLong(0);
  });

EXPORT(stencil_matrix_delete,{

    void* _stencil;
    if (!PyArg_ParseTuple(args, "l", &_stencil)) {
      return NULL;
    }
    
    cgpt_stencil_matrix_base* stencil = (cgpt_stencil_matrix_base*)_stencil;

    delete stencil;

    return PyLong_FromLong(0);
  });
