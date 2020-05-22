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
#include "fp16.h"

EXPORT(fp16_to_fp32,{
    PyObject* _src,* _dst;
    long nfloats_share_exponent;
    if (!PyArg_ParseTuple(args, "OOl", &_dst,&_src, &nfloats_share_exponent)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(_src) && PyMemoryView_Check(_dst));
    Py_buffer* buf_src = PyMemoryView_GET_BUFFER(_src);
    Py_buffer* buf_dst = PyMemoryView_GET_BUFFER(_dst);
    ASSERT(PyBuffer_IsContiguous(buf_src,'C'));
    ASSERT(PyBuffer_IsContiguous(buf_dst,'C'));
    uint16_t* s = (uint16_t*)buf_src->buf;
    long len_src = buf_src->len;

    //std::cout << GridLogMessage << "fp16_to_fp32: " << s << ", " << len_src << std::endl;

    float* d = (float*)buf_dst->buf;
    long len_dst = buf_dst->len;

    ASSERT(len_src % 2 == 0);
    len_src /= 2;

    ASSERT(len_src % (nfloats_share_exponent + 1) == 0);
    long sites = len_src / ( nfloats_share_exponent + 1 );

    ASSERT(sites * nfloats_share_exponent * 4 == len_dst);

    thread_for(idx,sites,{
	float max = unmap_fp16_exp(s[(nfloats_share_exponent+1)*idx]);
	float min = -max;
	
	for (int i=0;i<nfloats_share_exponent;i++) {
	  d[idx * nfloats_share_exponent + i] = 
	    fp_unmap( s[(nfloats_share_exponent+1)*idx + 1 + i], min, max, SHRT_UMAX );
	}
      });

    return PyLong_FromLong(0);
  });


EXPORT(fp32_to_fp16,{
    PyObject* _src,* _dst;
    long nfloats_share_exponent;
    if (!PyArg_ParseTuple(args, "OOl", &_dst,&_src, &nfloats_share_exponent)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(_src) && PyMemoryView_Check(_dst));
    Py_buffer* buf_src = PyMemoryView_GET_BUFFER(_src);
    Py_buffer* buf_dst = PyMemoryView_GET_BUFFER(_dst);
    ASSERT(PyBuffer_IsContiguous(buf_src,'C'));
    ASSERT(PyBuffer_IsContiguous(buf_dst,'C'));
    float* s = (float*)buf_src->buf;
    long len_src = buf_src->len;
    uint16_t* d = (uint16_t*)buf_dst->buf;
    long len_dst = buf_dst->len;

    ASSERT(len_src % 4 == 0);
    len_src /= 4;

    ASSERT(len_dst % 2 == 0);
    len_dst /= 2;

    ASSERT(len_dst % (nfloats_share_exponent + 1) == 0);
    long sites = len_dst / ( nfloats_share_exponent + 1 );
    ASSERT(sites * nfloats_share_exponent == len_src);

    thread_for(idx,sites,{
	// find max
	float max = fabs(s[idx*nfloats_share_exponent]);
	float min;
    
	for (int i=1;i<nfloats_share_exponent;i++) {
	  float x = fabs(s[idx*nfloats_share_exponent + i]);
	  if (x > max)
	    max = x;
	}

	unsigned short exp = map_fp16_exp(max);
	max = unmap_fp16_exp(exp);
	min = -max;
	
	d[idx*(nfloats_share_exponent+1)]=exp;

	for (int i=0;i<nfloats_share_exponent;i++) {
	  d[idx * (nfloats_share_exponent+1) + 1 + i] = 
	    fp_map( s[nfloats_share_exponent*idx + i], min, max, SHRT_UMAX );
	}
      });

    return PyLong_FromLong(0);
  });


EXPORT(mixed_fp32fp16_to_fp32,{
    PyObject* _src,* _dst;
    long nfloats_share_exponent,block_size_fp32,block_size_fp16;
    if (!PyArg_ParseTuple(args, "OOlll", &_dst,&_src, &block_size_fp32,&block_size_fp16,&nfloats_share_exponent)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(_src) && PyMemoryView_Check(_dst));
    Py_buffer* buf_src = PyMemoryView_GET_BUFFER(_src);
    Py_buffer* buf_dst = PyMemoryView_GET_BUFFER(_dst);
    ASSERT(PyBuffer_IsContiguous(buf_src,'C'));
    ASSERT(PyBuffer_IsContiguous(buf_dst,'C'));
    char* s = (char*)buf_src->buf;
    long len_src = buf_src->len;

    float* d = (float*)buf_dst->buf;
    long len_dst = buf_dst->len;

    long block_size = block_size_fp32 + block_size_fp16;

    ASSERT(len_src % block_size == 0);
    long blocks = len_src / block_size;

    ASSERT(block_size_fp16 % ((nfloats_share_exponent + 1)*2) == 0);
    long sites = block_size_fp16 / ( nfloats_share_exponent + 1 ) / 2;

    ASSERT(block_size_fp32 % 4 == 0);
    long floats_per_fp32_block = block_size_fp32 / 4;
    long floats_per_block = sites * nfloats_share_exponent + floats_per_fp32_block;

    ASSERT(blocks * (sites * nfloats_share_exponent * 4 + block_size_fp32) == len_dst);

    thread_for(b,blocks,{

	char* s_block = s + block_size * b;
	float* s_float = (float*)s_block;

	// first copy fp32
	memcpy(&d[b*floats_per_block],s_float,block_size_fp32);

	// then decompress fp16
	for (long idx=0;idx<sites;idx++) {

	  uint16_t* s_site = (uint16_t*)(s_block + block_size_fp32) + idx * (nfloats_share_exponent + 1);

	  float max = unmap_fp16_exp(s_site[0]);
	  float min = -max;
	  
	  for (int i=0;i<nfloats_share_exponent;i++) {
	    d[b*floats_per_block + floats_per_fp32_block + idx*nfloats_share_exponent + i] = 
	      fp_unmap( s_site[1 + i], min, max, SHRT_UMAX );
	  }
	}
      });

    return PyLong_FromLong(0);
  });



EXPORT(fp32_to_mixed_fp32fp16,{
    PyObject* _src,* _dst;
    long nfloats_share_exponent,block_size_fp32,block_size_fp16;
    if (!PyArg_ParseTuple(args, "OOlll", &_dst,&_src, &block_size_fp32,&block_size_fp16,&nfloats_share_exponent)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(_src) && PyMemoryView_Check(_dst));
    Py_buffer* buf_src = PyMemoryView_GET_BUFFER(_src);
    Py_buffer* buf_dst = PyMemoryView_GET_BUFFER(_dst);
    ASSERT(PyBuffer_IsContiguous(buf_src,'C'));
    ASSERT(PyBuffer_IsContiguous(buf_dst,'C'));
    float* s = (float*)buf_src->buf;
    long len_src = buf_src->len;

    char* d = (char*)buf_dst->buf;
    long len_dst = buf_dst->len;

    long block_size = block_size_fp32 + block_size_fp16;

    ASSERT(len_dst % block_size == 0);
    long blocks = len_dst / block_size;

    ASSERT(block_size_fp16 % ((nfloats_share_exponent + 1)*2) == 0);
    long sites = block_size_fp16 / ( nfloats_share_exponent + 1 ) / 2;

    ASSERT(block_size_fp32 % 4 == 0);
    long floats_per_fp32_block = block_size_fp32 / 4;
    long floats_per_block = sites * nfloats_share_exponent + floats_per_fp32_block;

    ASSERT(blocks * (sites * nfloats_share_exponent * 4 + block_size_fp32) == len_src);

    thread_for(b,blocks,{

	char* d_block = d + block_size * b;
	float* d_float = (float*)d_block;

	// first copy fp32
	memcpy(d_float,&s[b*floats_per_block],block_size_fp32);

	// then compress fp16
	for (long idx=0;idx<sites;idx++) {

	  uint16_t* d_site = (uint16_t*)(d_block + block_size_fp32) + idx * (nfloats_share_exponent + 1);

	  // find max
	  float max = fabs(s[b*floats_per_block + floats_per_fp32_block + idx*nfloats_share_exponent]);
	  float min;
	  
	  for (int i=1;i<nfloats_share_exponent;i++) {
	    float x = fabs(s[b*floats_per_block + floats_per_fp32_block + idx*nfloats_share_exponent + i]);
	    if (x > max)
	      max = x;
	  }

	  unsigned short exp = map_fp16_exp(max);
	  max = unmap_fp16_exp(exp);
	  min = -max;
	
	  d_site[0]=exp;

	  for (int i=0;i<nfloats_share_exponent;i++) {
	    d_site[1 + i] = 
	      fp_map( s[b*floats_per_block + floats_per_fp32_block + idx*nfloats_share_exponent + i] , min, max, SHRT_UMAX );
	  }
	}
      });

    return PyLong_FromLong(0);
  });


