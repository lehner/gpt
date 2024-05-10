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
    long _code_parallel_block_size;
    long _local;
    if (!PyArg_ParseTuple(args, "llOOll", &_lattice, &_grid, &_shifts, &_code,
			  &_code_parallel_block_size, &_local)) {
      return NULL;
    }
    
    GridBase* grid = (GridBase*)_grid;
    cgpt_Lattice_base* lattice = (cgpt_Lattice_base*)_lattice;

    return PyLong_FromVoidPtr(lattice->stencil_matrix(grid, _shifts, _code,
						      _code_parallel_block_size,
						      _local));
  });

EXPORT(stencil_matrix_vector_create,{

    void* _grid;
    void* _lattice_matrix;
    void* _lattice_vector;
    PyObject* _shifts, * _code;
    long _code_parallel_block_size;
    long _local;
    if (!PyArg_ParseTuple(args, "lllOOll", &_lattice_matrix, &_lattice_vector, &_grid, &_shifts, &_code,
			  &_code_parallel_block_size, &_local)) {
      return NULL;
    }
    
    GridBase* grid = (GridBase*)_grid;
    cgpt_Lattice_base* lattice_matrix = (cgpt_Lattice_base*)_lattice_matrix;
    cgpt_Lattice_base* lattice_vector = (cgpt_Lattice_base*)_lattice_vector;

    return PyLong_FromVoidPtr(lattice_vector->stencil_matrix_vector(lattice_matrix, grid, _shifts, _code,
								    _code_parallel_block_size,
								    _local));
  });

EXPORT(stencil_tensor_create,{

    void* _grid;
    void* _lattice;
    PyObject* _shifts, * _code, * _segments;
    long _code_parallel_block_size;
    long _local;
    if (!PyArg_ParseTuple(args, "llOOOl", &_lattice, &_grid, &_shifts, &_code,
			  &_segments, &_local)) {
      return NULL;
    }
    
    GridBase* grid = (GridBase*)_grid;
    cgpt_Lattice_base* lattice = (cgpt_Lattice_base*)_lattice;

    return PyLong_FromVoidPtr(lattice->stencil_tensor(grid, _shifts, _code,
						      _segments, _local));
  });

EXPORT(stencil_matrix_execute,{

    void* _stencil;
    PyObject* _fields;
    long fast_osites;
    if (!PyArg_ParseTuple(args, "lOl", &_stencil, &_fields, &fast_osites)) {
      return NULL;
    }
    
    cgpt_stencil_matrix_base* stencil = (cgpt_stencil_matrix_base*)_stencil;

    std::vector<cgpt_Lattice_base*> __fields;
    cgpt_basis_fill(__fields,_fields);

    stencil->execute(__fields, fast_osites);

    return PyLong_FromLong(0);
  });


EXPORT(stencil_matrix_vector_execute,{

    void* _stencil;
    PyObject* _matrix_fields;
    PyObject* _vector_fields;
    long fast_osites;
    if (!PyArg_ParseTuple(args, "lOOl", &_stencil, &_matrix_fields, &_vector_fields, &fast_osites)) {
      return NULL;
    }
    
    cgpt_stencil_matrix_vector_base* stencil = (cgpt_stencil_matrix_vector_base*)_stencil;

    std::vector<cgpt_Lattice_base*> __matrix_fields, __vector_fields;
    cgpt_basis_fill(__matrix_fields,_matrix_fields);
    cgpt_basis_fill(__vector_fields,_vector_fields);

    stencil->execute(__matrix_fields, __vector_fields, fast_osites);

    return PyLong_FromLong(0);
  });

EXPORT(stencil_tensor_execute,{

    void* _stencil;
    PyObject* _fields;
    long osites_per_instruction;
    long osites_per_cache_block;
    if (!PyArg_ParseTuple(args, "lOll", &_stencil, &_fields,
			  &osites_per_instruction,
			  &osites_per_cache_block)) {
      return NULL;
    }
    
    cgpt_stencil_tensor_base* stencil = (cgpt_stencil_tensor_base*)_stencil;

    std::vector<cgpt_Lattice_base*> __fields;
    cgpt_basis_fill(__fields,_fields);

    cgpt_stencil_tensor_execute_params_t params =
      {
	(int)osites_per_instruction,
	(int)osites_per_cache_block
      };
    stencil->execute(__fields, params);

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

EXPORT(stencil_matrix_vector_delete,{

    void* _stencil;
    if (!PyArg_ParseTuple(args, "l", &_stencil)) {
      return NULL;
    }
    
    cgpt_stencil_matrix_vector_base* stencil = (cgpt_stencil_matrix_vector_base*)_stencil;

    delete stencil;

    return PyLong_FromLong(0);
  });

EXPORT(stencil_tensor_delete,{

    void* _stencil;
    if (!PyArg_ParseTuple(args, "l", &_stencil)) {
      return NULL;
    }
    
    cgpt_stencil_tensor_base* stencil = (cgpt_stencil_tensor_base*)_stencil;

    delete stencil;

    return PyLong_FromLong(0);
  });
