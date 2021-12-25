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
#include "munge.h"

// swap axes
EXPORT(munge_inner_outer,{
    PyObject* _src,* _dst;
    long inner, outer;
    if (!PyArg_ParseTuple(args, "OOll", &_dst,&_src, &inner, &outer)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(_src) && PyMemoryView_Check(_dst));
    Py_buffer* buf_src = PyMemoryView_GET_BUFFER(_src);
    Py_buffer* buf_dst = PyMemoryView_GET_BUFFER(_dst);
    ASSERT(PyBuffer_IsContiguous(buf_src,'C'));
    ASSERT(PyBuffer_IsContiguous(buf_dst,'C'));
    char* s = (char*)buf_src->buf;
    char* d = (char*)buf_dst->buf;
    long len_src = buf_src->len;
    if (len_src > 0) {
      ASSERT(len_src == buf_dst->len);
      ASSERT(len_src % (inner*outer) == 0);
      long blocksize = len_src / (inner*outer);

      thread_for(o,outer,{

	  for (long i=0;i<inner;i++) {
	    memcpy(d + (i*outer + o)*blocksize,s + (o*inner  + i)*blocksize,blocksize);
	  }
	});

    }
    return PyLong_FromLong(0);
  });

EXPORT(munge_byte_order,{
    PyObject* _src,* _dst;
    long word_size;
    if (!PyArg_ParseTuple(args, "OOl", &_dst,&_src,&word_size)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(_src) && PyMemoryView_Check(_dst));
    Py_buffer* buf_src = PyMemoryView_GET_BUFFER(_src);
    Py_buffer* buf_dst = PyMemoryView_GET_BUFFER(_dst);
    ASSERT(PyBuffer_IsContiguous(buf_src,'C'));
    ASSERT(PyBuffer_IsContiguous(buf_dst,'C'));
    char* s = (char*)buf_src->buf;
    char* d = (char*)buf_dst->buf;
    long len_src = buf_src->len;
    if (len_src > 0) {
      ASSERT(len_src == buf_dst->len);
      ASSERT(len_src % word_size == 0);
      long words = len_src / word_size;

      thread_for(w,words,{
	  char* w_s = &s[word_size*w];
	  char* w_d = &d[word_size*w];
	  char buf[word_size];
	  for (long byte=0;byte<word_size;byte++) {
	    buf[byte] = w_s[word_size - 1 - byte];
	  }
	  memcpy(w_d,buf,word_size);
	});

    }
    return PyLong_FromLong(0);
  });

EXPORT(munge_reconstruct_third_row,{
    // reconstructs third row in a unitary matrix
    PyObject* _src,* _dst;
    long float_size;
    if (!PyArg_ParseTuple(args, "OOl", &_dst,&_src,&float_size)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(_src) && PyMemoryView_Check(_dst));
    Py_buffer* buf_src = PyMemoryView_GET_BUFFER(_src);
    Py_buffer* buf_dst = PyMemoryView_GET_BUFFER(_dst);
    ASSERT(PyBuffer_IsContiguous(buf_src,'C'));
    ASSERT(PyBuffer_IsContiguous(buf_dst,'C'));
    char* s = (char*)buf_src->buf;
    char* d = (char*)buf_dst->buf;
    long len_src = buf_src->len;
    long len_dst = buf_dst->len;
    ASSERT(len_dst * 2 == len_src * 3);
    if (len_src > 0) {

      long blocks = len_src / (3*2*float_size*2);

      if (float_size == 8) {
        munger_reconstruct_third_row((ComplexD*)d, (ComplexD*)s, blocks);
      } else if (float_size == 4) {
        munger_reconstruct_third_row((ComplexF*)d, (ComplexF*)s, blocks);
      } else {
        ERR("Unknown float_size = %ld", float_size);
      }

    }
    return PyLong_FromLong(0);
  });
