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

EXPORT(create_lookup_table,{

    void* _grid_c,*_mask;
    if (!PyArg_ParseTuple(args, "ll", &_grid_c, &_mask)) {
      return NULL;
    }

    GridBase* grid_c = (GridBase*)_grid_c;
    cgpt_Lattice_base* mask = (cgpt_Lattice_base*)_mask;

    return PyLong_FromVoidPtr((void*) mask->create_lookup_table(grid_c, mask));
  });

EXPORT(delete_lookup_table,{

    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }

    delete ((cgpt_lookup_table_base*)p);
    return PyLong_FromLong(0);
  });
