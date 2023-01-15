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

std::map<std::string,_grid_cache_entry_> cgpt_grid_cache;
std::map<GridBase*,std::string> cgpt_grid_cache_tag;

EXPORT(create_grid,{
    
    PyObject* _fdimensions, * _precision, * _cb_mask, * _simd_mask, * _mpi;
    void* _parent_grid;
    if (!PyArg_ParseTuple(args, "OOOOOl", &_fdimensions, &_precision,&_cb_mask,&_simd_mask,&_mpi,&_parent_grid)) {
      return NULL;
    }
    
    Coordinate fdimensions, mpi, cb_mask, simd_mask;
    GridBase* parent_grid = (GridBase*)_parent_grid;
    std::string precision;
    
    cgpt_convert(_fdimensions,fdimensions);
    cgpt_convert(_mpi,mpi);
    cgpt_convert(_precision,precision);
    cgpt_convert(_cb_mask,cb_mask);
    cgpt_convert(_simd_mask,simd_mask);
    
    int Nsimd;
    if (precision == "single") {
      Nsimd = vComplexF::Nsimd();
    } else if (precision == "double") {
      Nsimd = vComplexD::Nsimd();
    } else {
      ERR("Unknown precision");
    }

    long nd = cb_mask.size();
    ASSERT(nd == fdimensions.size());
    ASSERT(nd == simd_mask.size());
    ASSERT(nd == mpi.size());
    Coordinate simd(nd,1);

    int nn=Nsimd;
    int i=nd-1;
    while (nn > 1) {
      if (simd_mask[i]) {
	simd[i]*=2;
	nn/=2;
      }
      i--;
      if (i<0)
	i=nd-1;
    }

    GridBase* grid;

    // graceful exit if grid not compatible with mpi/simd
    for (long d=0;d<nd;d++) {
      int cb_factor = cb_mask[d] ? 2 : 1;

      if (fdimensions[d] % (simd[d] * mpi[d] * cb_factor) != 0) {
	ERR("Dimension %d is not consistent:\n"
	    " fdimension = %d\n"
	    " simd = %d\n"
	    " mpi = %d\n"
	    " cb = %d\n",d,fdimensions[d],simd[d],mpi[d],cb_factor);
      }
    }

    grid = cgpt_create_grid(fdimensions,simd,cb_mask,mpi,parent_grid);
    return PyLong_FromVoidPtr(grid);
    
  });



EXPORT(delete_grid,{
    
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    cgpt_delete_grid((GridBase*)p);
    return PyLong_FromLong(0);
    
  });



EXPORT(grid_barrier,{
    
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    ((GridBase*)p)->Barrier();
    return PyLong_FromLong(0);
    
  });
  


EXPORT(grid_globalsum,{
    
    void* p;
    PyObject* o;
    if (!PyArg_ParseTuple(args, "lO", &p,&o)) {
      return NULL;
    }
    
    GridBase* grid = (GridBase*)p;
    if (PyComplex_Check(o)) {
      ComplexD c;
      cgpt_convert(o,c);
      grid->GlobalSum(c);
      return PyComplex_FromDoubles(c.real(),c.imag());
    } else if (PyFloat_Check(o)) {
      RealD c;
      cgpt_convert(o,c);
      grid->GlobalSum(c);
      return PyFloat_FromDouble(c);
    } else if (PyLong_Check(o)) {
      uint64_t c;
      cgpt_convert(o,c);
      grid->GlobalSum(c);
      return PyLong_FromLong(c);
    } else if (cgpt_PyArray_Check(o)) {
      PyArrayObject* ao = (PyArrayObject*)o;
      int dt = PyArray_TYPE(ao);
      void* data = PyArray_DATA(ao);
      size_t nbytes = PyArray_NBYTES(ao);
      if (dt == NPY_FLOAT32 || dt == NPY_COMPLEX64) {
	grid->GlobalSumVector((RealF*)data, nbytes / 4);
      } else if (dt == NPY_FLOAT64 || dt == NPY_COMPLEX128) {
	grid->GlobalSumVector((RealD*)data, nbytes / 8);
      } else if (dt == NPY_UINT64) {
	grid->GlobalSumVector((uint64_t*)data, nbytes / 8);
      } else {
	ERR("Unsupported numy data type (single, double, csingle, cdouble, uint64 currently allowed)");
      }
      Py_XINCREF(o);
      return o;
    } else {
      ERR("Unsupported object");
    }
  });

EXPORT(grid_get_processor,{
    
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    GridBase* grid = (GridBase*)p;
    long srank,sranks;
    cgpt_grid_get_info(grid,srank,sranks);
    long rank = grid->_processor;
    long ranks = grid->_Nprocessors;
    PyObject* coor = cgpt_convert(grid->_processor_coor);
    PyObject* ldims = cgpt_convert(grid->_ldimensions);
    PyObject* gdims = cgpt_convert(grid->_gdimensions);
    return Py_BuildValue("(l,l,N,N,N,l,l)",rank,ranks,coor,gdims,ldims,srank,sranks);
    
  });
 
EXPORT(grid_broadcast,{

    long root;
    PyObject* _data;
    void* p;
    if (!PyArg_ParseTuple(args, "llO", &p,&root,&_data)) {
      return NULL;
    }

    GridBase* grid = (GridBase*)p;

    ASSERT(cgpt_PyArray_Check(_data));
    PyArrayObject* data = (PyArrayObject*)_data;
    
    char* data_p = (char*)PyArray_DATA(data);
    
    long sz = (long)PyArray_NBYTES(data);

    ASSERT(sz < INT_MAX);

    grid->Broadcast((int)root, data_p, (int)sz);
    Py_INCREF(Py_None);
    return Py_None;

  });

EXPORT(grid_exchange,{

    long send_to, recv_from;
    PyObject* _send_data, * _recv_data;
    void* p;
    if (!PyArg_ParseTuple(args, "lllOO", &p,&send_to,&recv_from,&_send_data,&_recv_data)) {
      return NULL;
    }

    GridBase* grid = (GridBase*)p;

    ASSERT(cgpt_PyArray_Check(_send_data) && cgpt_PyArray_Check(_recv_data));
    PyArrayObject* send_data = (PyArrayObject*)_send_data;
    PyArrayObject* recv_data = (PyArrayObject*)_recv_data;
    
    char* send_p = (char*)PyArray_DATA(send_data);
    char* recv_p = (char*)PyArray_DATA(recv_data);
    
    long sz_send = (long)PyArray_NBYTES(send_data);
    long sz_recv = (long)PyArray_NBYTES(recv_data);

    ASSERT(sz_send == sz_recv);
    ASSERT(sz_send < INT_MAX);

    grid->SendToRecvFrom(send_p, (int)send_to, recv_p, (int)recv_from, sz_send);

    Py_INCREF(Py_None);
    return Py_None;

  });

