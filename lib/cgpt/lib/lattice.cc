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
#include <set>

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

void lattice_fill_types(std::set<std::string> & types) {
#define INSTANTIATE(v,t,n) types.insert(get_otype(n<v>()));
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

EXPORT(lattice_types,{
    std::set<std::string> types;
    lattice_fill_types(types);
    
    PyObject* list = PyList_New(types.size());
    long i = 0;
    for (auto & t : types) {
      PyList_SetItem(list,i++,PyUnicode_FromString(t.c_str()));
    }
    
    return list;
    
  });

EXPORT(lattice_set_to_number,{
    void* p;
    PyObject* _number;
    if (!PyArg_ParseTuple(args, "lO", &p, &_number)) {
      return NULL;
    }

    ComplexD number;
    cgpt_convert(_number,number);
    ((cgpt_Lattice_base*)p)->set_to_number(number);

    return PyLong_FromLong(0);
  });

EXPORT(lattice_set_to_identity,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }

    ((cgpt_Lattice_base*)p)->set_to_identity();

    return PyLong_FromLong(0);
  });

EXPORT(lattice_memory_view,{
    void* p;
    PyObject* _lattice_view_location;
    PyObject* _lattice;
    if (!PyArg_ParseTuple(args, "OlO", &_lattice, &p, &_lattice_view_location)) {
      return NULL;
    }

    std::string lattice_view_location;
    cgpt_convert(_lattice_view_location,lattice_view_location);
    memory_type lattice_view_mt = cgpt_memory_type_from_string(lattice_view_location);
    
    cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
    PyObject* r = l->memory_view(lattice_view_mt);

    struct _mvc {
      PyObject* _lattice;
      PyObject* _base;
    };

    _mvc* pm = new _mvc;
    pm->_lattice = _lattice;
    Py_INCREF(_lattice);
    pm->_base = ((PyMemoryViewObject*)r)->mbuf->master.obj;
    
    PyObject *capsule =
      PyCapsule_New((void*)pm, NULL,
		    [] (PyObject *capsule) -> void { 
		      _mvc* pm = (_mvc*)PyCapsule_GetPointer(capsule, NULL);
		      Py_DECREF(pm->_base);
		      Py_DECREF(pm->_lattice);
		      delete pm;
		    });

    ((PyMemoryViewObject*)r)->mbuf->master.obj = capsule;

    return r;
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
