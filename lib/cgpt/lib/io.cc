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
#include "io/common.h"
#include "io/nersc.h"
#include "io/openQCD.h"

EXPORT(load,{
    PyObject* ret;

    // nersc gauge configuration
    if ((ret = load_nersc(args)))
      return ret;

    // openQCD file format is minimal, not distinctive, test last
    if ((ret = load_openQCD(args)))
      return ret;

    return Py_BuildValue("(OO)",Py_None,Py_None);
    
  });

EXPORT(save,{

    std::string dest, format;
    bool verbose;
    PyObject* _dest,* _objs,* _format,* _verbose;
    if (!PyArg_ParseTuple(args, "OOOO", &_dest, &_objs, &_format, &_verbose)) {
      return NULL;
    }

    cgpt_convert(_dest,dest);
    cgpt_convert((PyObject*)_format->ob_type,format);
    cgpt_convert(_verbose,verbose);

    if (format == "nersc") {
      save_nersc(dest, _format, _objs, verbose);
    } else {
      ERR("Unknown format: %s", format.c_str());
    }
    
    Py_RETURN_NONE;
    
  });

EXPORT(fopen,{
    PyObject* _fn,* _md;
    std::string fn, md;
    if (!PyArg_ParseTuple(args, "OO", &_fn,&_md)) {
      return NULL;
    }
    cgpt_convert(_fn,fn);
    cgpt_convert(_md,md);

    return PyLong_FromVoidPtr(fopen(fn.c_str(),md.c_str()));
  });

EXPORT(fclose,{
    void* _file;
    if (!PyArg_ParseTuple(args, "l", &_file)) {
      return NULL;
    }
    fclose((FILE*)_file);
    return PyLong_FromLong(0);
  });

EXPORT(ftell,{
    void* _file;
    if (!PyArg_ParseTuple(args, "l", &_file)) {
      return NULL;
    }
    return PyLong_FromLong((long)ftello((FILE*)_file));
  });

EXPORT(fflush,{
    void* _file;
    if (!PyArg_ParseTuple(args, "l", &_file)) {
      return NULL;
    }
    return PyLong_FromLong((long)fflush((FILE*)_file));
  });

EXPORT(fseek,{
    void* _file;
    long offset, _whence;
    int whence;
    if (!PyArg_ParseTuple(args, "lll", &_file,&offset,&_whence)) {
      return NULL;
    }
    switch(_whence) {
    case 0:
      whence=SEEK_SET;
      break;
    case 1:
      whence=SEEK_CUR;
      break;
    case 2:
      whence=SEEK_END;
      break;
    default:
      ERR("Unknown seek whence");
    }
    return PyLong_FromLong((long)fseeko((FILE*)_file,offset,whence));
  });

EXPORT(fread,{
    void* _file;
    long size;
    PyObject* dst;
    if (!PyArg_ParseTuple(args, "llO", &_file,&size,&dst)) {
      return NULL;
    }
    ASSERT(PyMemoryView_Check(dst));
    Py_buffer* buf = PyMemoryView_GET_BUFFER(dst);
    ASSERT(PyBuffer_IsContiguous(buf,'C'));
    void* d = buf->buf;
    long len = buf->len;
    ASSERT(len >= size);
    return PyLong_FromLong(fread(d,size,1,(FILE*)_file));
  });

EXPORT(fwrite,{
    void* _file;
    long size;
    PyObject* src;
    if (!PyArg_ParseTuple(args, "llO", &_file,&size,&src)) {
      return NULL;
    }
    ASSERT(PyMemoryView_Check(src));
    Py_buffer* buf = PyMemoryView_GET_BUFFER(src);
    ASSERT(PyBuffer_IsContiguous(buf,'C'));
    void* s = buf->buf;
    long len = buf->len;
    ASSERT(len >= size);
    return PyLong_FromLong(fwrite(s,size,1,(FILE*)_file));
  });
