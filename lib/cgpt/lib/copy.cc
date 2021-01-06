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

EXPORT(copy_view_embeded_in_communicator,{
    void* p,* _grid;
    if (!PyArg_ParseTuple(args, "ll", &p,&_grid)) {
      return NULL;
    }
    
    cgpt_gm_view* v = (cgpt_gm_view*)p;
    GridBase* grid = (GridBase*)_grid;
    return PyLong_FromVoidPtr(cgpt_view_embeded_in_communicator(v,grid));
  });

EXPORT(copy_view_add_index_offset,{
    void* p;
    long offset;
    if (!PyArg_ParseTuple(args, "ll", &p,&offset)) {
      return NULL;
    }
    
    cgpt_gm_view* v = (cgpt_gm_view*)p;
    cgpt_view_index_offset(v->view, (uint32_t)offset);
    return PyLong_FromLong(0);
  });

EXPORT(copy_add_views,{
    void* _a, *_b;
    if (!PyArg_ParseTuple(args, "ll", &_a, &_b)) {
      return NULL;
    }
    
    cgpt_gm_view* a = (cgpt_gm_view*)_a;
    cgpt_gm_view* b = (cgpt_gm_view*)_b;
    return PyLong_FromVoidPtr(cgpt_add_views(a,b));
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
    long local_only, skip_optimize;
    
    if (!PyArg_ParseTuple(args, "llOll", &_vdst, &_vsrc, &_tbuffer, &local_only, &skip_optimize)) {
      return NULL;
    }

    cgpt_convert(_tbuffer,tbuffer);

    cgpt_gm_view* vsrc = (cgpt_gm_view*)_vsrc;
    cgpt_gm_view* vdst = (cgpt_gm_view*)_vdst;

    ASSERT(vsrc->comm == vdst->comm);
    ASSERT(vsrc->rank == vdst->rank);
    
    gm_transfer* plan = new gm_transfer(vsrc->rank, vsrc->comm);
    memory_type mt = cgpt_memory_type_from_string(tbuffer);

    plan->create(vdst->view, vsrc->view, mt,local_only,skip_optimize);

    return PyLong_FromVoidPtr(plan);
  });

EXPORT(copy_get_plan_info,{
    long _plan;
    if (!PyArg_ParseTuple(args, "l", &_plan)) {
      return NULL;
    }

    gm_transfer* plan = (gm_transfer*)_plan;

    PyObject* ret = PyDict_New();
    for (auto & rank : plan->blocks) {
      auto rank_dst = rank.first.first;
      auto rank_src = rank.first.second;

      PyObject* ret_rank = PyDict_New();
      for (auto & index : rank.second) {
	auto index_dst = index.first.first;
	auto index_src = index.first.second;
	size_t blocks = index.second.second.size();
	size_t size = index.second.first * blocks;

	PyObject* data = PyDict_New();
	PyDict_SetItemString(data,"blocks",PyLong_FromLong((long)blocks));
	PyDict_SetItemString(data,"size",PyLong_FromLong((long)size));
	PyDict_SetItem(ret_rank, Py_BuildValue("(ll)",index_dst, index_src), data);
      }

      PyDict_SetItem(ret, Py_BuildValue("(ll)",rank_dst, rank_src), ret_rank);
    }

    return ret;
  });
    
EXPORT(copy_execute_plan,{

    long _plan;
    PyObject* _dst,* _src,* _lattice_view_location;
    std::string lattice_view_location;
    
    if (!PyArg_ParseTuple(args, "lOOO", &_plan, &_dst, &_src, &_lattice_view_location)) {
      return NULL;
    }

    cgpt_convert(_lattice_view_location, lattice_view_location);
    memory_type lattice_view_mt = cgpt_memory_type_from_string(lattice_view_location);

    gm_transfer* plan = (gm_transfer*)_plan;

    std::vector<gm_transfer::memory_view> vdst, vsrc;

    std::vector<PyObject*> lattice_views;

    cgpt_copy_add_memory_views(vdst, _dst, lattice_views, lattice_view_mt);
    cgpt_copy_add_memory_views(vsrc, _src, lattice_views, lattice_view_mt);

    plan->execute(vdst, vsrc);

    for (auto v : lattice_views)
      Py_XDECREF(v);

    return PyLong_FromVoidPtr(0);
  });

EXPORT(copy_cyclic_upscale,{
    PyObject* input;
    long sz_target;
    
    if (!PyArg_ParseTuple(args, "Ol", &input, &sz_target)) {
      return NULL;
    }

    // for now only support arrays, in the future may also support memoryviews
    if (cgpt_PyArray_Check(input)) {
      return cgpt_copy_cyclic_upscale_array((PyArrayObject*)input, (size_t)sz_target);
    }

    Py_XINCREF(input);
    return input;
  });

EXPORT(copy_create_view_from_lattice,{
    PyObject* pos, * vlat, * tidx;
    
    if (!PyArg_ParseTuple(args, "OOO", &vlat, &pos, &tidx)) {
      return NULL;
    }

    ASSERT(cgpt_PyArray_Check(pos));
    ASSERT(cgpt_PyArray_Check(tidx));

    cgpt_gm_view* v = new cgpt_gm_view();
    GridBase* grid = cgpt_copy_append_view_from_vlattice(v->view,vlat,0,1,(PyArrayObject*)pos,(PyArrayObject*)tidx);

    v->comm = grid->communicator;
    v->rank = grid->_processor;
    
    return PyLong_FromVoidPtr(v);
  });

EXPORT(copy_create_view,{

    long _grid;
    PyObject* _a;
    
    if (!PyArg_ParseTuple(args, "lO", &_grid, &_a)) {
      return NULL;
    }

    GridBase* grid = (GridBase*)_grid;
    cgpt_gm_view* v = new cgpt_gm_view();

    if (grid) {
      v->comm = grid->communicator;
      v->rank = grid->_processor;
    } else {
      v->comm = CartesianCommunicator::communicator_world;
      v->rank = CartesianCommunicator::RankWorld();
    }

    long nb;
    int64_t* ad;

    if (_a == Py_None) {
      nb = 0;
      ad = 0;
    } else {
      ASSERT(cgpt_PyArray_Check(_a));
      PyArrayObject* a = (PyArrayObject*)_a;

      ASSERT(PyArray_NDIM(a) == 2);
      long* tdim = PyArray_DIMS(a);
      nb = tdim[0];
      ASSERT(tdim[1] == 4);
      ASSERT(PyArray_TYPE(a)==NPY_INT64);
      ad = (int64_t*)PyArray_DATA(a);
    }

    struct {
      int64_t* ad;
      long nb;
      
      long operator[](size_t i) const {
	return (long)ad[4 * i + 3];
      }
      
      size_t size() const {
	return nb;
      }
    } get_block_size;

    get_block_size.ad = ad;
    get_block_size.nb = nb;

    long block_size = cgpt_reduce(get_block_size, cgpt_gcd, nb ? get_block_size[0] : 1);
    ASSERT(block_size);

    v->view.block_size = block_size;

    std::vector<gm_view::block_t> blocks;
    blocks.resize(nb);
    std::vector<size_t> ndup(nb);
    thread_for(i, nb, {
	auto & x = blocks[i];
	x.rank = (int)ad[4 * i + 0];
	if (x.rank < 0)
	  x.rank = v->rank;
	x.index = (uint32_t)ad[4 * i + 1];
	x.start = (uint64_t)ad[4 * i + 2];
	ndup[i] = (long)ad[4 * i + 3] / block_size;
      });

    cgpt_duplicate(v->view.blocks, blocks, ndup,
		   [block_size](size_t index, size_t sub_index, gm_view::block_t & b) {
		     b.start += sub_index * block_size;
		   });

    return PyLong_FromVoidPtr(v);
  });

