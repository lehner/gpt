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

EXPORT(create_block_map,{

    void* _grid_c,*_mask;
    long  basis_virtual_size, basis_n_block;
    PyObject* _basis;
    if (!PyArg_ParseTuple(args, "lOlll", &_grid_c, &_basis, &basis_virtual_size, &basis_n_block, &_mask)) {
      return NULL;
    }

    GridBase* grid_c = (GridBase*)_grid_c;
    cgpt_Lattice_base* mask = (cgpt_Lattice_base*)_mask;

    std::vector<cgpt_Lattice_base*> basis;
    long basis_n_virtual = cgpt_basis_fill(basis,_basis);

    ASSERT(basis.size() > 0);

    return PyLong_FromVoidPtr((void*) basis[0]->block_map(grid_c, basis, basis_n_virtual, basis_virtual_size, basis_n_block, mask));
  });

EXPORT(delete_block_map,{

    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }

    delete ((cgpt_block_map_base*)p);
    return PyLong_FromLong(0);
  });

EXPORT(block_project,{

    void* _map;
    PyObject * _coarse, * _fine;
    if (!PyArg_ParseTuple(args, "lOO", &_map, &_coarse,&_fine)) {
      return NULL;
    }

    cgpt_block_map_base* map = (cgpt_block_map_base*)_map;

    std::vector<cgpt_Lattice_base*> fine;
    long fine_n_virtual = cgpt_basis_fill(fine, _fine);

    std::vector<cgpt_Lattice_base*> coarse;
    long coarse_n_virtual = cgpt_basis_fill(coarse, _coarse);

    map->project(coarse, coarse_n_virtual,
		 fine, fine_n_virtual);

    return PyLong_FromLong(0);
  });

EXPORT(block_promote,{

    void* _map;
    PyObject * _coarse, * _fine;
    if (!PyArg_ParseTuple(args, "lOO", &_map, &_coarse,&_fine)) {
      return NULL;
    }

    cgpt_block_map_base* map = (cgpt_block_map_base*)_map;

    std::vector<cgpt_Lattice_base*> fine;
    long fine_n_virtual = cgpt_basis_fill(fine, _fine);

    std::vector<cgpt_Lattice_base*> coarse;
    long coarse_n_virtual = cgpt_basis_fill(coarse, _coarse);

    map->promote(coarse, coarse_n_virtual, 
		 fine, fine_n_virtual);

    return PyLong_FromLong(0);
  });

EXPORT(block_orthonormalize,{

    void* _map;
    if (!PyArg_ParseTuple(args, "l", &_map)) {
      return NULL;
    }

    cgpt_block_map_base* map = (cgpt_block_map_base*)_map;

    map->orthonormalize();

    return PyLong_FromLong(0);
  });

EXPORT(block_sum,{

    void* _map;
    PyObject * _coarse, * _fine;
    if (!PyArg_ParseTuple(args, "lOO", &_map, &_coarse,&_fine)) {
      return NULL;
    }

    cgpt_block_map_base* map = (cgpt_block_map_base*)_map;

    std::vector<cgpt_Lattice_base*> fine;
    long fine_n_virtual = cgpt_basis_fill(fine, _fine);

    std::vector<cgpt_Lattice_base*> coarse;
    long coarse_n_virtual = cgpt_basis_fill(coarse, _coarse);

    map->sum(coarse, coarse_n_virtual, 
	     fine, fine_n_virtual);

    return PyLong_FromLong(0);
  });

EXPORT(block_embed,{

    void* _map;
    PyObject * _coarse, * _fine;
    if (!PyArg_ParseTuple(args, "lOO", &_map, &_coarse,&_fine)) {
      return NULL;
    }

    cgpt_block_map_base* map = (cgpt_block_map_base*)_map;

    std::vector<cgpt_Lattice_base*> fine;
    long fine_n_virtual = cgpt_basis_fill(fine, _fine);

    std::vector<cgpt_Lattice_base*> coarse;
    long coarse_n_virtual = cgpt_basis_fill(coarse, _coarse);

    map->embed(coarse, coarse_n_virtual, 
	       fine, fine_n_virtual);

    return PyLong_FromLong(0);
  });
