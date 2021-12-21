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

EXPORT(util_ferm2prop,{
    
    void* _ferm,* _prop;
    long spin, color;
    PyObject* _f2p;
    if (!PyArg_ParseTuple(args, "llllO", &_ferm, &_prop, &spin, &color, &_f2p)) {
      return NULL;
    }
    
    cgpt_Lattice_base* ferm = (cgpt_Lattice_base*)_ferm;
    cgpt_Lattice_base* prop = (cgpt_Lattice_base*)_prop;
    
    bool f2p;
    cgpt_convert(_f2p,f2p);
    
    ferm->ferm_to_prop(prop,(int)spin,(int)color,f2p);
    
    return PyLong_FromLong(0);
  });

EXPORT(util_crc32,{
    
    PyObject* _mem;
    long crc32_prev;
    if (!PyArg_ParseTuple(args, "Ol", &_mem,&crc32_prev)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(_mem));
    Py_buffer* buf = PyMemoryView_GET_BUFFER(_mem);
    ASSERT(PyBuffer_IsContiguous(buf,'C'));
    unsigned char* data = (unsigned char*)buf->buf;
    int64_t len = (int64_t)buf->len;

    uint32_t crc = cgpt_crc32(data,len,(uint32_t)crc32_prev);
    
    return PyLong_FromLong(crc);
  });

EXPORT(util_crc32_combine,{
    
    long crc32_prev, crc32;
    long size;
    if (!PyArg_ParseTuple(args, "lll", &crc32_prev, &crc32, &size)) {
      return NULL;
    }
    return PyLong_FromLong(crc32_combine(crc32_prev,crc32,size));
  });

EXPORT(util_nersc_checksum,{
    
    PyObject* _mem;
    long cs_prev;
    if (!PyArg_ParseTuple(args, "Ol", &_mem,&cs_prev)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(_mem));
    Py_buffer* buf = PyMemoryView_GET_BUFFER(_mem);
    ASSERT(PyBuffer_IsContiguous(buf,'C'));
    unsigned char* data = (unsigned char*)buf->buf;
    int64_t len = (int64_t)buf->len;

    uint32_t crc = cgpt_nersc_checksum(data,len,(uint32_t)cs_prev);
    
    return PyLong_FromLong(crc);
  });

EXPORT(util_sha256,{
    
    PyObject* _mem;
    if (!PyArg_ParseTuple(args, "O", &_mem)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(_mem));
    Py_buffer* buf = PyMemoryView_GET_BUFFER(_mem);
    ASSERT(PyBuffer_IsContiguous(buf,'C'));
    unsigned char* data = (unsigned char*)buf->buf;
    int64_t len = (int64_t)buf->len;

    uint32_t sha256[8];
    cgpt_sha256(sha256,data,len);
    
    return PyTuple_Pack(8,
			PyLong_FromLong(sha256[0]),
			PyLong_FromLong(sha256[1]),
			PyLong_FromLong(sha256[2]),
			PyLong_FromLong(sha256[3]),
			PyLong_FromLong(sha256[4]),
			PyLong_FromLong(sha256[5]),
			PyLong_FromLong(sha256[6]),
			PyLong_FromLong(sha256[7])
                        );
  });

EXPORT(util_mem,{

    size_t accelerator_available = 0x0;
    size_t accelerator_total = 0x0;

#ifdef GRID_CUDA
    cudaMemGetInfo(&accelerator_available,&accelerator_total);
#endif

    // TODO: add more information out of MemoryManager::
    //static uint64_t     DeviceBytes;
    //static uint64_t     DeviceLRUBytes;
    //static uint64_t     DeviceMaxBytes;
    //static uint64_t     HostToDeviceBytes;
    //static uint64_t     DeviceToHostBytes;
    //static uint64_t     HostToDeviceXfer;
    //static uint64_t     DeviceToHostXfer;

    return Py_BuildValue("{s:k,s:k}",
			 "accelerator_available", (unsigned long)accelerator_available,
			 "accelerator_total", (unsigned long)accelerator_total);
  });

EXPORT(mview,{

    // default python memoryview(ndarray()) too slow, below faster
    PyObject* _a;
    if (!PyArg_ParseTuple(args, "O", &_a)) {
      return NULL;
    }

    if (cgpt_PyArray_Check(_a)) {
      char* data = (char*)PyArray_DATA((PyArrayObject*)_a);
      long nbytes = PyArray_NBYTES((PyArrayObject*)_a);
      PyObject* r = PyMemoryView_FromMemory(data,nbytes,PyBUF_WRITE);
      Py_XINCREF(_a);
      ASSERT(!((PyMemoryViewObject*)r)->mbuf->master.obj);
      ((PyMemoryViewObject*)r)->mbuf->master.obj = _a;
      return r;
    } else {
      ERR("Unsupported type");
    }

    return NULL;

  });

EXPORT(ndarray,{
    
    PyObject* _dim;
    PyObject* _dtype;
    if (!PyArg_ParseTuple(args, "OO", &_dim, &_dtype)) {
      return NULL;
    }
    
    std::vector<long> dim;
    cgpt_convert(_dim, dim);

    int dtype;
    cgpt_convert(_dtype,dtype);
    
    return (PyObject*)cgpt_new_PyArray((long)dim.size(), &dim[0], (int)dtype);
    
  });
