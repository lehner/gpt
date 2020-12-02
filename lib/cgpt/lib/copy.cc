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
#include "copy.h"

EXPORT(copy_delete_view,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    cgpt_gm_view* v = (cgpt_gm_view*)p;
    delete v;
    return PyLong_FromLong(0);
  });

EXPORT(copy_view_size,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    cgpt_gm_view* v = (cgpt_gm_view*)p;
    return PyLong_FromLong((long)v->view.size());
  });

EXPORT(copy_delete_plan,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    gm_transfer* v = (gm_transfer*)p;
    delete v;
    return PyLong_FromLong(0);
  });

EXPORT(copy_create_plan,{

    long _vsrc, _vdst;
    PyObject* _tbuffer;
    std::string tbuffer;
    
    if (!PyArg_ParseTuple(args, "llO", &_vsrc, &_vdst, &_tbuffer)) {
      return NULL;
    }

    cgpt_convert(_tbuffer,tbuffer);

    cgpt_gm_view* vsrc = (cgpt_gm_view*)_vsrc;
    cgpt_gm_view* vdst = (cgpt_gm_view*)_vdst;

    ASSERT(vsrc->comm == vdst->comm);
    ASSERT(vsrc->rank == vdst->rank);
    
    gm_transfer* plan = new gm_transfer(vsrc->rank, vsrc->comm);
    memory_type mt = cgpt_memory_type_from_string(tbuffer);
    plan->create(vdst->view, vsrc->view, mt);

    return PyLong_FromVoidPtr(plan);
  });

EXPORT(copy_execute_plan,{

    long _plan;
    PyObject* _dst,* _src;
    
    if (!PyArg_ParseTuple(args, "lOO", &_plan, &_dst, &_src)) {
      return NULL;
    }

    gm_transfer* plan = (gm_transfer*)_plan;

    std::vector<gm_transfer::memory_view> vdst, vsrc;

    std::vector<PyObject*> lattice_views;

    cgpt_copy_add_memory_views(vdst, _dst, lattice_views);
    cgpt_copy_add_memory_views(vsrc, _src, lattice_views);
    
    plan->execute(vdst, vsrc);

    for (auto v : lattice_views)
      Py_XDECREF(v);

    return PyLong_FromVoidPtr(0);
  });

EXPORT(copy_create_view_from_lattice,{
    PyObject* pos, * vlat, * tidx;
    
    if (!PyArg_ParseTuple(args, "OOO", &vlat, &pos, &tidx)) {
      return NULL;
    }

    ASSERT(cgpt_PyArray_Check(pos));
    ASSERT(cgpt_PyArray_Check(tidx));

    cgpt_gm_view* v = new cgpt_gm_view();
    GridBase* grid = append_view_from_vlattice(v->view,vlat,0,1,(PyArrayObject*)pos,(PyArrayObject*)tidx);

    v->comm = grid->communicator;
    v->rank = grid->_processor;
    
    return PyLong_FromVoidPtr(v);
  });

/*
  Add views
*/

/*
  To global communicator

      v->comm = CartesianCommunicator::communicator_world;
      v->rank = CartesianCommunicator::RankWorld();

      std::vector<uint64_t> rank_map(grid->_Nprocessors,0);
      rank_map[grid->_processor] = (uint64_t)v->rank;
      grid->GlobalSumVector(&rank_map[0],rank_map.size());

      thread_for(i, v->view.blocks.size(), {
	  auto & x = v->view.blocks[i];
	  x.rank = (int)rank_map[grid->_processor];
	});
    }
*/

EXPORT(copy_create_view,{

    long _grid;
    PyObject* _a;
    
    if (!PyArg_ParseTuple(args, "lO", &_grid, &_a)) {
      return NULL;
    }

    ASSERT(cgpt_PyArray_Check(_a));
    PyArrayObject* a = (PyArrayObject*)_a;

    ASSERT(PyArray_NDIM(a) == 2);
    long* tdim = PyArray_DIMS(a);
    long nb = tdim[0];
    ASSERT(tdim[1] == 4);
    ASSERT(PyArray_TYPE(a)==NPY_INT64);
    int64_t* ad = (int64_t*)PyArray_DATA(a);

    GridBase* grid = (GridBase*)_grid;
    cgpt_gm_view* v = new cgpt_gm_view();

    if (grid) {
      v->comm = grid->communicator;
      v->rank = grid->_processor;
    } else {
      v->comm = CartesianCommunicator::communicator_world;
      v->rank = CartesianCommunicator::RankWorld();
    }

    v->view.blocks.resize(nb);
    thread_for(i, nb, {
	  auto & x = v->view.blocks[i];
	  x.rank = (int)ad[4 * i + 0];
	  if (x.rank < 0)
	    x.rank = v->rank;
	  x.index = (uint32_t)ad[4 * i + 1];
	  x.start = (uint64_t)ad[4 * i + 2];
	  x.size = (uint64_t)ad[4 * i + 3];
	});

    return PyLong_FromVoidPtr(v);
  });

