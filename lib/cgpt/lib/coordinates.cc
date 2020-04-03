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

EXPORT(coordinates_form_cartesian_view,{

    PyObject* _top,* _bottom;
    if (!PyArg_ParseTuple(args, "OO", &_top, &_bottom)) {
      return NULL;
    }

    ASSERT(PyList_Check(_top) && PyList_Check(_bottom));
    std::vector<long> top, bottom, dims;
    std::vector<int32_t> size;
    cgpt_convert(_top,top);
    cgpt_convert(_bottom,bottom);
    int Nd = (int)top.size();
    ASSERT(top.size() == bottom.size());
    long points = 1;
    for (int i=0;i<Nd;i++) {
      ASSERT(bottom[i] >= top[i]);
      size.push_back((int32_t)(bottom[i] - top[i]));
      points *= size[i];
    }
    dims.push_back(points);
    dims.push_back(Nd);

    PyArrayObject* a = (PyArrayObject*)PyArray_SimpleNew((int)dims.size(), &dims[0], NPY_INT32);
    int32_t* d = (int32_t*)PyArray_DATA(a);
    thread_region {
      std::vector<int32_t> coor(Nd);
      thread_for_in_region(idx,points,{
	  Lexicographic::CoorFromIndex(coor,idx,size);
	  for (int i=0;i<Nd;i++)
	    d[Nd*idx + i] = top[i] + coor[i];
	});
    }
    return (PyObject*)a;
  });

EXPORT(mview,{

    PyObject* _a;
    if (!PyArg_ParseTuple(args, "O", &_a)) {
      return NULL;
    }

    if (PyArray_Check(_a)) {
      char* data = (char*)PyArray_DATA((PyArrayObject*)_a);
      long nbytes = PyArray_NBYTES((PyArrayObject*)_a);
      PyObject* r = PyMemoryView_FromMemory(data,nbytes,PyBUF_WRITE);
      Py_XINCREF(_a);
      PyMemoryView_GET_BASE(r) = _a;
      return r;
    } else {
      ERR("Unsupported type");
    }

    return NULL;

  });
