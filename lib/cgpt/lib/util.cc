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
    if (!PyArg_ParseTuple(args, "O", &_mem)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(_mem));
    Py_buffer* buf = PyMemoryView_GET_BUFFER(_mem);
    ASSERT(PyBuffer_IsContiguous(buf,'C'));
    unsigned char* data = (unsigned char*)buf->buf;
    int64_t len = (int64_t)buf->len;

    // crc32 of zlib was incorrect for very large sizes, so do it block-wise
    uint32_t crc = 0x0;
    off_t blk = 0;
    off_t step = 1024*1024*1024;
    while (len > step) {
      crc = crc32(crc,&data[blk],step);
      blk += step;
      len -= step;
    }

    crc = crc32(crc,&data[blk],len);
    
    return PyLong_FromLong(crc);
  });
