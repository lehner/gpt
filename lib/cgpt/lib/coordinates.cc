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

    PyObject* _top,* _bottom,* _checker_dim_mask,* _cb;
    if (!PyArg_ParseTuple(args, "OOOO", &_top, &_bottom,&_checker_dim_mask,&_cb)) {
      return NULL;
    }

    ASSERT(PyList_Check(_top) && PyList_Check(_bottom));
    std::vector<long> top, bottom, dims, checker_dim_mask;
    std::vector<int32_t> size;
    cgpt_convert(_top,top);
    cgpt_convert(_bottom,bottom);
    cgpt_convert(_checker_dim_mask,checker_dim_mask);
    int Nd = (int)top.size();
    ASSERT(Nd == bottom.size());
    ASSERT(Nd == checker_dim_mask.size());
    long points = 1;
    for (int i=0;i<Nd;i++) {
      ASSERT(bottom[i] >= top[i]);
      size.push_back((int32_t)(bottom[i] - top[i]));
      points *= size[i];
    }

    int cb, cbf, cbd;
    long fstride = 1;
    if (Py_None == _cb) {
      cb=0;
      cbf=1;
      cbd=0;
    } else if (PyLong_Check(_cb)) {
      cb=(int)PyLong_AsLong(_cb);
      cbf=2;
      for (cbd=0;cbd<Nd;cbd++) {
	if (checker_dim_mask[cbd])
	  break;
	fstride *= size[cbd];
      }
    } else {
      ERR("Unknown argument type for _cb");
    }

    //std::cout << GridLogMessage << top << ", " << bottom << ", " << checker_dim_mask << ": " << cb << ", " << cbf << ", " << fstride << std::endl;

    dims.push_back(points/cbf);
    dims.push_back(Nd);

    PyArrayObject* a = (PyArrayObject*)PyArray_SimpleNew((int)dims.size(), &dims[0], NPY_INT32);
    int32_t* d = (int32_t*)PyArray_DATA(a);
    thread_region 
      {
	std::vector<int32_t> coor(Nd);
	thread_for_in_region(idx,points,{
	    Lexicographic::CoorFromIndex(coor,idx,size);
	    long idx_cb = (idx % fstride) + ((idx / fstride)/cbf) * fstride;
	    long site_cb = 0;
	    for (int i=0;i<Nd;i++)
	      if (checker_dim_mask[i])
		site_cb += top[i] + coor[i];
	    if (site_cb % 2 == cb) {
	      for (int i=0;i<Nd;i++)
		d[Nd*idx_cb + i] = top[i] + coor[i];
	    }
	  });
      }

    // xoxo
    // oxox

    PyArray_CLEARFLAGS(a,NPY_ARRAY_WRITEABLE); // read-only, so we can cache distribute plans
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
