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
#include "io/ildg.h"
#include "io/openQCD.h"

EXPORT(load,{
    PyObject* ret;

    // nersc gauge configuration
    if ((ret = load_nersc(args)))
      return ret;

    // ildg gauge configuration
    if ((ret = load_ildg(args)))
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


struct cgpt_FILE {
  int fd;
  bool write;
  off_t pos;
};

EXPORT(fopen,{
    PyObject* _fn,* _md;
    std::string fn, md;
    if (!PyArg_ParseTuple(args, "OO", &_fn,&_md)) {
      return NULL;
    }
    cgpt_convert(_fn,fn);
    cgpt_convert(_md,md);

    bool binary = (md.find("b") != std::string::npos);
    bool read = (md.find("r") != std::string::npos);
    bool write = (md.find("w") != std::string::npos);
    bool append = (md.find("a") != std::string::npos);
    bool readwrite = (md.find("+") != std::string::npos);

    ASSERT(_FILE_OFFSET_BITS >=  64);
    ASSERT(sizeof(off_t) >= 8);

    cgpt_FILE* file = new cgpt_FILE();

    int flags = 0;

    if (write || append)
      flags |= O_CREAT;

    if (readwrite) {
      flags |= O_RDWR;
      file->write = true;
    } else if (read) {
      flags |= O_RDONLY;
      file->write = false;
    } else if (write) {
      flags |= O_WRONLY|O_TRUNC;
      file->write = true;
    } else if (append) {
      flags |= O_WRONLY;
      file->write = true;
    } else {
      ERR("Neither read nor write in %s", md.c_str());
    }

    if (append)
      flags |= O_APPEND;

    file->fd = open(fn.c_str(), flags, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    file->pos = 0;

    if (append) {
      file->pos = lseek(file->fd, 0, SEEK_END);
      if (file->pos == (off_t)-1) {
	ERR("Could not determine file offset in append mode for %s", fn.c_str());
      }
    }
      
    if (file->fd == -1) {
      delete file;
      file = 0;
    }
    
    return PyLong_FromVoidPtr(file);
  });

EXPORT(fclose,{
    cgpt_FILE* _file;
    if (!PyArg_ParseTuple(args, "l", &_file)) {
      return NULL;
    }

    int err;
    if (_file->write) {
      while ((err = fsync(_file->fd)) == EINTR);

      if (err)
	ERR("fsync failed during fclose: %d\n", err);
    }
    
    err = close(_file->fd);
    if (err)
      ERR("fclose failed: %d\n", err);

    delete _file;
    
    return PyLong_FromLong(0);
  });

EXPORT(ftell,{
    cgpt_FILE* _file;
    if (!PyArg_ParseTuple(args, "l", &_file)) {
      return NULL;
    }
    return PyLong_FromLong((long)_file->pos);
  });

EXPORT(fseek,{
    cgpt_FILE* _file;
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

    _file->pos = lseek(_file->fd, offset, whence);
    if (_file->pos == (off_t)-1)
      return PyLong_FromLong(-1);
    else
      return PyLong_FromLong(0);
  });

EXPORT(fread,{
    cgpt_FILE* _file;
    long size;
    PyObject* dst;
    if (!PyArg_ParseTuple(args, "llO", &_file,&size,&dst)) {
      return NULL;
    }
    ASSERT(PyMemoryView_Check(dst));
    Py_buffer* buf = PyMemoryView_GET_BUFFER(dst);
    ASSERT(PyBuffer_IsContiguous(buf,'C'));
    char* d = (char*)buf->buf;
    long len = buf->len;
    ASSERT(len >= size);

    while (size > 0) {
      size_t n = read(_file->fd, d, size);
      if (n == (size_t)-1) {
	if (errno == EAGAIN || errno == EINTR)
	  continue;
	else
	  ERR("Error in read of %ld bytes: %d\n", (long)size, errno);
      }

      if (n == 0) {
	return PyLong_FromLong(0);
      }

      ASSERT(n <= size);
      
      d += n;
      size -= n;
      _file->pos += n;
    }
    
    return PyLong_FromLong(1);
  });

EXPORT(fwrite,{
    cgpt_FILE* _file;
    long size;
    PyObject* src;
    if (!PyArg_ParseTuple(args, "llO", &_file,&size,&src)) {
      return NULL;
    }
    ASSERT(PyMemoryView_Check(src));
    Py_buffer* buf = PyMemoryView_GET_BUFFER(src);
    ASSERT(PyBuffer_IsContiguous(buf,'C'));
    char* s = (char*)buf->buf;
    long len = buf->len;
    ASSERT(len >= size);

    while (size > 0) {
      size_t n = write(_file->fd, s, size);
      if (n == (size_t)-1) {
	if (errno == EAGAIN || errno == EINTR)
	  continue;
	else
	  ERR("Error in read of %ld bytes: %d\n", (long)size, errno);
      }

      ASSERT(n <= size);
      
      s += n;
      size -= n;
      _file->pos += n;
    }
    
    return PyLong_FromLong(1);
  });
