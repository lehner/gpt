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

typedef void* (* create_lattice_prec_otype)(GridBase* grid);
std::map<std::string,create_lattice_prec_otype> _create_otype_;
std::map<std::string,int> _otype_singlet_rank_;

#define INSTANTIATE(v,t,n) void lattice_init_ ## t ## _ ## n();
#include "instantiate/instantiate.h"
#undef INSTANTIATE

void lattice_init() {
#define INSTANTIATE(v,t,n) lattice_init_ ## t ## _ ## n();
#include "instantiate/instantiate.h"
#undef INSTANTIATE
}

EXPORT(create_lattice,{

    void* _grid;
    PyObject* _otype, * _prec;
    if (!PyArg_ParseTuple(args, "lOO", &_grid, &_otype, &_prec)) {
      return NULL;
    }
    
    GridBase* grid = (GridBase*)_grid;
    std::string otype;
    std::string prec;
    
    cgpt_convert(_otype,otype);
    cgpt_convert(_prec,prec);
    
    void* plat = 0;
    std::string tag = prec + ":" + otype;
    auto f = _create_otype_.find(tag);
    if (f == _create_otype_.end()) {
      ERR("Unknown field type: %s, %s", otype.c_str(), prec.c_str());
    }
    
    return PyLong_FromVoidPtr(f->second(grid));
  });

EXPORT(delete_lattice,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    delete ((cgpt_Lattice_base*)p);
    return PyLong_FromLong(0);
  });

EXPORT(lattice_set_to_zero,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    ((cgpt_Lattice_base*)p)->set_to_zero();

    return PyLong_FromLong(0);
  });
  
EXPORT(lattice_memory_view,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
    return l->memory_view(mt_host);
  });

EXPORT(lattice_export,{
    PyObject* pos, * vlat, * tidx,* _shape;
    if (!PyArg_ParseTuple(args, "OOOO", &vlat, &pos, &tidx, &_shape)) {
      return NULL;
    }

    ASSERT(cgpt_PyArray_Check(pos));
    ASSERT(cgpt_PyArray_Check(tidx));

    std::vector<long> shape;
    cgpt_convert(_shape,shape);

    gm_view src, dst;
    GridBase* grid = append_view_from_vlattice(src,vlat,0,1,(PyArrayObject*)pos,(PyArrayObject*)tidx);

    PyArrayObject* dst_array = create_array_to_hold_view(dst,src,vlat,shape,grid->_processor);

    gm_transfer plan(grid->_processor, grid->communicator);

    plan.create(dst, src, mt_host);

    std::vector<gm_transfer::memory_view> vdst, vsrc;

    append_memory_view_from_dense_array(vdst,dst_array);

    std::vector<PyObject*> views;
    append_memory_view_from_vlat(vsrc,vlat,mt_host,views);

    plan.execute(vdst,vsrc);

    for (auto v : views)
      Py_XDECREF(v);
    
    return (PyObject*)dst_array;
  });

EXPORT(lattice_import,{
    PyObject* pos, *vlat, * d, * tidx;
    if (!PyArg_ParseTuple(args, "OOOO", &vlat, &pos, &tidx, &d)) {
      return NULL;
    }

    ASSERT(cgpt_PyArray_Check(pos));
    ASSERT(cgpt_PyArray_Check(tidx));

    gm_view src, dst;
    GridBase* grid = append_view_from_vlattice(dst,vlat,0,1,(PyArrayObject*)pos,(PyArrayObject*)tidx);

    size_t sz_dst = dst.size();

    if (cgpt_PyArray_Check(d)) {
      d = append_view_from_dense_array(src,(PyArrayObject*)d,sz_dst,grid->_processor);

      gm_transfer plan(grid->_processor, grid->communicator);

      plan.create(dst, src, mt_host);

      std::vector<gm_transfer::memory_view> vdst, vsrc;
      
      append_memory_view_from_dense_array(vsrc,(PyArrayObject*)d);
      
      std::vector<PyObject*> views;
      append_memory_view_from_vlat(vdst,vlat,mt_host,views);
      
      plan.execute(vdst,vsrc);
      
      for (auto v : views)
	Py_XDECREF(v);
      
      Py_XDECREF(d);

    } else if (PyMemoryView_Check(d)) {

      append_view_from_memory_view(src,d, grid->_processor);

      gm_transfer plan(grid->_processor, grid->communicator);

      plan.create(dst, src, mt_host);
      
      std::vector<gm_transfer::memory_view> vdst, vsrc;
      
      append_memory_view_from_memory_view(vsrc,d);

    
      std::vector<PyObject*> views;
      append_memory_view_from_vlat(vdst,vlat,mt_host,views);
      
      plan.execute(vdst,vsrc);
      
      for (auto v : views)
	Py_XDECREF(v);


    } else if (d == Py_None) {
      
      gm_transfer plan(grid->_processor, grid->communicator);

      plan.create(dst, src, mt_host);
      
      std::vector<gm_transfer::memory_view> vdst, vsrc;
      
      append_memory_view_from_memory_view(vsrc,d);

      std::vector<PyObject*> views;
      append_memory_view_from_vlat(vdst,vlat,mt_host,views);
      
      plan.execute(vdst,vsrc);
      
      for (auto v : views)
	Py_XDECREF(v);

    } else {
      ERR("Unknown import data");
    }

    return PyLong_FromLong(0);
  });

EXPORT(lattice_to_str,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
    return l->to_str();
  });

EXPORT(lattice_pick_checkerboard,{
    void* _src, *_dst;
    long cb;
    if (!PyArg_ParseTuple(args, "lll", &cb, &_src,&_dst)) {
      return NULL;
    }
    cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    dst->pick_checkerboard_from(cb == 0 ? Even : Odd, src);
    
    return PyLong_FromLong(0);
  });

EXPORT(lattice_set_checkerboard,{
    void* _src, *_dst;
    if (!PyArg_ParseTuple(args, "ll", &_src,&_dst)) {
      return NULL;
    }
    cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    dst->set_checkerboard_from(src);
    
    return PyLong_FromLong(0);
  });

EXPORT(lattice_change_checkerboard,{
    void* _dst;
    long cb;
    if (!PyArg_ParseTuple(args, "ll", &_dst,&cb)) {
      return NULL;
    }
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    dst->change_checkerboard(cb == 0 ? Even : Odd);
    
    return PyLong_FromLong(0);
  });

EXPORT(lattice_get_checkerboard,{
    void* _dst;
    if (!PyArg_ParseTuple(args, "l", &_dst)) {
      return NULL;
    }
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    return PyLong_FromLong(dst->get_checkerboard() == Even ? 0 : 1);
  });

EXPORT(lattice_advise,{
    void* _dst;
    PyObject* _type;
    if (!PyArg_ParseTuple(args, "lO", &_dst,&_type)) {
      return NULL;
    }
    std::string type;
    cgpt_convert(_type,type);
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    return dst->advise(type);
  });

EXPORT(lattice_prefetch,{
    void* _dst;
    PyObject* _type;
    if (!PyArg_ParseTuple(args, "lO", &_dst,&_type)) {
      return NULL;
    }
    std::string type;
    cgpt_convert(_type,type);
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    return dst->prefetch(type);
  });
