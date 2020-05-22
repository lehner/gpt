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
#include "coordinates.h"

EXPORT(coordinates_from_cartesian_view,{

    PyObject* _top,* _bottom,* _checker_dim_mask,* _cb,* _order;
    if (!PyArg_ParseTuple(args, "OOOOO", &_top, &_bottom,&_checker_dim_mask,&_cb,&_order)) {
      return NULL;
    }

    std::string order;
    cgpt_convert(_order,order);
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

    if (order == "grid") {

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

    } else if (order == "canonical") {

      Coordinate c_size = toCanonical(size);

      thread_region 
	{
	  Coordinate c_coor(Nd), coor(Nd);
	  thread_for_in_region(idx,points,{
	      Lexicographic::CoorFromIndex(c_coor,idx,c_size);
	      coor = fromCanonical(c_coor);

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

    } else {
      ERR("Unknown order scheme: %s",order.c_str());
    }

    // xoxo
    // oxox

    PyArray_CLEARFLAGS(a,NPY_ARRAY_WRITEABLE); // read-only, so we can cache distribute plans
    return (PyObject*)a;
  });

EXPORT(coordinates_from_block,{

    PyObject* _top,* _bottom,* _nb,* _type;
    long block;
    if (!PyArg_ParseTuple(args, "OOlOO", &_top, &_bottom,&block,&_nb,&_type)) {
      return NULL;
    }

    ASSERT(PyList_Check(_top) && PyList_Check(_bottom) && PyList_Check(_nb));
    std::vector<long> top, bottom, nb, block_size, dims, size;
    std::string type;
    cgpt_convert(_top,top);
    cgpt_convert(_bottom,bottom);
    cgpt_convert(_nb,nb);
    cgpt_convert(_type,type);
    int Nd = (int)top.size();
    ASSERT(Nd == bottom.size());
    ASSERT(Nd == nb.size());
    long points = 1;
    for (int i=0;i<Nd;i++) {
      ASSERT(bottom[i] >= top[i]);
      size.push_back(bottom[i] - top[i]);
      ASSERT(size[i] % nb[i] == 0);
      block_size.push_back(size[i] / nb[i]);
      points *= block_size[i];
    }

    if (type == "canonicalOdd") {

      Coordinate c_nb = toCanonical(nb);
      Coordinate c_block_size = toCanonical(block_size);
      Coordinate c_top = toCanonical(top);

      dims.push_back(points/2);
      dims.push_back(Nd);

      Coordinate c_block_coor(Nd), c_block_top(Nd);
      Lexicographic::CoorFromIndex(c_block_coor,block,c_nb);
      for (int i=0;i<Nd;i++)
	c_block_top[i] = c_top[i] + c_block_coor[i] * c_block_size[i];
      
      PyArrayObject* a = (PyArrayObject*)PyArray_SimpleNew((int)dims.size(), &dims[0], NPY_INT32);
      int32_t* d = (int32_t*)PyArray_DATA(a);
      
      if (Nd == 5) {
	thread_region 
	  {
	    Coordinate c_coor(Nd), c_g_coor(Nd), g_coor(Nd);
	    thread_for_in_region(idx,points,{
		Lexicographic::CoorFromIndex(c_coor,idx,c_block_size);
		for (int i=0;i<Nd;i++)
		  c_g_coor[i] = c_block_top[i] + c_coor[i];
		long site_sum = c_g_coor[0] + c_g_coor[1] + c_g_coor[2] + c_g_coor[3];
		long idx_cb = idx / 2;
		if (site_sum % 2 == 1) {
		  g_coor = fromCanonical(c_g_coor);
		  for (int i=0;i<Nd;i++)
		    d[Nd*idx_cb + i] = g_coor[i];
		}
	      });
	  }
      } else {
	ERR("Dimension %d not supported",Nd);
      }
      
      PyArray_CLEARFLAGS(a,NPY_ARRAY_WRITEABLE); // read-only, so we can cache distribute plans
      return (PyObject*)a;
      
    } else {
      ERR("Unknown type %s",type.c_str());
    }
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
