/*
    GPT - Grid Python Toolkit
    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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
#include "blas.h"

EXPORT(create_blas,{
    return PyLong_FromVoidPtr((void*)new cgpt_blas());
  });

EXPORT(delete_blas,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    delete ((cgpt_blas*)p);
    return PyLong_FromLong(0);
  });

EXPORT(blas_gemm,{
    void* p;
    long m,n,k,oA,oB,oC;
    PyObject* _alpha, *_beta, *vA, *vB, *vC, *iA, *iB, *iC, *_dtype;

    if (!PyArg_ParseTuple(args, "llllOOOlOOlOOOO", &p,&m,&n,&k,&_alpha,
			  &vA, &iA, &oA,
			  &vB, &iB, &oB,
			  &_beta,
			  &vC, &iC,&_dtype)) {
      return NULL;
    }

    ComplexD alpha, beta;
    cgpt_convert(_alpha,alpha);
    cgpt_convert(_beta,beta);

    ASSERT(PyMemoryView_Check(vA) && PyMemoryView_Check(vB) && PyMemoryView_Check(vC));
    ASSERT(PyArray_Check(iA) && PyArray_Check(iB) && PyArray_Check(iC));

    PyArrayObject* _iA = (PyArrayObject*)iA;
    PyArrayObject* _iB = (PyArrayObject*)iB;
    PyArrayObject* _iC = (PyArrayObject*)iC;
    
    ASSERT(PyArray_TYPE(_iA)==NPY_INT64);
    ASSERT(PyArray_TYPE(_iB)==NPY_INT64);
    ASSERT(PyArray_TYPE(_iC)==NPY_INT64);

    long nA = PyArray_SIZE(_iA);
    long nB = PyArray_SIZE(_iB);
    long nC = PyArray_SIZE(_iC);

    int64_t* idxA = (int64_t*)PyArray_DATA(_iA);
    int64_t* idxB = (int64_t*)PyArray_DATA(_iB);
    int64_t* idxC = (int64_t*)PyArray_DATA(_iC);

    ASSERT(nA == nB && nA == nC);

    Py_buffer* bufA = PyMemoryView_GET_BUFFER(vA);
    Py_buffer* bufB = PyMemoryView_GET_BUFFER(vB);
    Py_buffer* bufC = PyMemoryView_GET_BUFFER(vC);
    
    ASSERT(PyBuffer_IsContiguous(bufA,'C'));
    ASSERT(PyBuffer_IsContiguous(bufB,'C'));
    ASSERT(PyBuffer_IsContiguous(bufC,'C'));
    
    void* data_A = bufA->buf;
    void* data_B = bufB->buf;
    void* data_C = bufC->buf;
    
    size_t size_A = bufA->len;
    size_t size_B = bufB->len;
    size_t size_C = bufC->len;

    ASSERT(PyType_Check(_dtype));
    const char* __dtype = ((PyTypeObject*)_dtype)->tp_name;

    if (!strcmp(__dtype,"numpy.complex64")) {
      ((cgpt_blas*)p)->jobs.push_back(new cgpt_gemm_job<ComplexF>(m,n,k,alpha,data_A, idxA, oA,data_B,idxB, oB,beta,data_C,idxC,nC));
    } else if (!strcmp(__dtype,"numpy.complex128")) {
      ((cgpt_blas*)p)->jobs.push_back(new cgpt_gemm_job<ComplexD>(m,n,k,alpha,data_A, idxA, oA,data_B,idxB, oB,beta,data_C,idxC,nC));
    } else {
      ERR("Unknown dtype = %s\n", __dtype);
    }
    return PyLong_FromLong(0);
  });

EXPORT(blas_execute,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    ((cgpt_blas*)p)->execute();
    
    return PyLong_FromLong(0);
  });
