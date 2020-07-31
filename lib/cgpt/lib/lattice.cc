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
    return l->memory_view();
  });

EXPORT(lattice_memory_view_coordinates,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
    return l->memory_view_coordinates();
  });

EXPORT(lattice_export,{
    PyObject* pos, * vlat, * tidx,* _shape;
    if (!PyArg_ParseTuple(args, "OOOO", &vlat, &pos, &tidx, &_shape)) {
      return NULL;
    }

    ASSERT(PyArray_Check(pos));
    ASSERT(PyArray_Check(tidx));
    std::vector<cgpt_distribute::data_simd> data;
    std::vector<long> shape;
    GridBase* grid;
    int cb,dt;

    cgpt_prepare_vlattice_importexport(vlat,data,shape,(PyArrayObject*)tidx,grid,cb,dt);
    cgpt_convert(_shape,shape);

    return (PyObject*)cgpt_importexport(grid,cb,dt,data,shape,(PyArrayObject*)pos,0);
  });

EXPORT(lattice_import,{
    PyObject* pos, *vlat, * d, * tidx;
    if (!PyArg_ParseTuple(args, "OOOO", &vlat, &pos, &tidx, &d)) {
      return NULL;
    }

    ASSERT(PyArray_Check(pos));
    ASSERT(PyArray_Check(tidx));
    std::vector<cgpt_distribute::data_simd> data;
    std::vector<long> shape;
    GridBase* grid;
    int cb,dt;

    cgpt_prepare_vlattice_importexport(vlat,data,shape,(PyArrayObject*)tidx,grid,cb,dt);
    cgpt_importexport(grid,cb,dt,data,shape,(PyArrayObject*)pos,d);

    return PyLong_FromLong(0);
  });

EXPORT(lattice_import_view,{
    PyObject * vlat_dst, *vlat_src, * pos_dst, * tidx_dst, * pos_src, * tidx_src;
    if (!PyArg_ParseTuple(args, "OOOOOO", &vlat_dst, &pos_dst, &tidx_dst, &vlat_src, &pos_src, &tidx_src)) {
      return NULL;
    }

    ASSERT(PyArray_Check(pos_dst) && PyArray_Check(pos_src));
    ASSERT(PyArray_Check(tidx_dst) && PyArray_Check(tidx_src));
    std::vector<cgpt_distribute::data_simd> data_dst, data_src;
    std::vector<long> shape_dst, shape_src;
    GridBase* grid_dst,* grid_src;
    int cb_dst, dt_dst, cb_src, dt_src;

    cgpt_prepare_vlattice_importexport(vlat_dst,data_dst,shape_dst,(PyArrayObject*)tidx_dst,grid_dst,cb_dst,dt_dst);
    cgpt_prepare_vlattice_importexport(vlat_src,data_src,shape_src,(PyArrayObject*)tidx_src,grid_src,cb_src,dt_src);

    cgpt_importexport(grid_dst,grid_src,
		      cb_dst,cb_src,
		      data_dst,data_src,
		      (PyArrayObject*)pos_dst,
		      (PyArrayObject*)pos_src);

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
