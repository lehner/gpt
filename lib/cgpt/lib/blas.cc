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

EXPORT(blas_accumulate,{
    long n;
    void* p;
    PyObject* v, *_dtype;

    if (!PyArg_ParseTuple(args, "llOO", &p, &n, &v, &_dtype)) {
      return NULL;
    }

    ASSERT(PyList_Check(v));
    size_t n_buf = PyList_Size(v);
    std::vector<void*> a_data(n_buf);
    
    for (int _i=0;_i<n_buf;_i++) {
      PyObject* _v = PyList_GetItem(v,_i);

      ASSERT(PyMemoryView_Check(_v));

      Py_buffer* bufA = PyMemoryView_GET_BUFFER(_v);
      ASSERT(PyBuffer_IsContiguous(bufA,'C'));

      a_data[_i] = bufA->buf;
    }

    ASSERT(PyType_Check(_dtype));
    const char* __dtype = ((PyTypeObject*)_dtype)->tp_name;

    if (!strcmp(__dtype,"numpy.complex64")) {
      ((cgpt_blas*)p)->jobs.push_back(new cgpt_accumulate_job<ComplexF>(n,a_data));
    } else if (!strcmp(__dtype,"numpy.complex128")) {
      ((cgpt_blas*)p)->jobs.push_back(new cgpt_accumulate_job<ComplexD>(n,a_data));
    } else {
      ERR("Unknown dtype = %s\n", __dtype);
    }
    return PyLong_FromLong(0);
  });

EXPORT(blas_gemm,{
    void* p;
    long m,n,k,oA,oB,oC;
    PyObject* _alpha, *_beta, *vA, *vB, *vC, *iA, *iB, *iC, *_dtype, *_precision;

    if (!PyArg_ParseTuple(args, "llllOOOlOOlOOOOO", &p,&m,&n,&k,&_alpha,
			  &vA, &iA, &oA,
			  &vB, &iB, &oB,
			  &_beta,
			  &vC, &iC,&_dtype,
			  &_precision)) {
      return NULL;
    }

    ComplexD alpha, beta;
    std::string precision; // compute precision
    cgpt_convert(_alpha,alpha);
    cgpt_convert(_beta,beta);
    cgpt_convert(_precision,precision);

    ASSERT(PyList_Check(vA) && PyList_Check(vB) && PyList_Check(vC));
    ASSERT(PyList_Check(iA) && PyList_Check(iB) && PyList_Check(iC));

    size_t n_buf = PyList_Size(vA);

    std::vector<void*> a_data_A(n_buf), a_data_B(n_buf), a_data_C(n_buf);
    std::vector<int64_t*> a_idxA(n_buf), a_idxB(n_buf), a_idxC(n_buf);
    std::vector<long> a_n(n_buf);
    
    for (int _i=0;_i<n_buf;_i++) {
      PyObject* _vA = PyList_GetItem(vA,_i);
      PyObject* _vB = PyList_GetItem(vB,_i);
      PyObject* _vC = PyList_GetItem(vC,_i);
      PyObject* _iA = PyList_GetItem(iA,_i);
      PyObject* _iB = PyList_GetItem(iB,_i);
      PyObject* _iC = PyList_GetItem(iC,_i);

      ASSERT(PyMemoryView_Check(_vA) && PyMemoryView_Check(_vB) && PyMemoryView_Check(_vC));
      ASSERT(PyArray_Check(_iA) && PyArray_Check(_iB) && PyArray_Check(_iC));

      PyArrayObject* __iA = (PyArrayObject*)_iA;
      PyArrayObject* __iB = (PyArrayObject*)_iB;
      PyArrayObject* __iC = (PyArrayObject*)_iC;
      
      ASSERT(PyArray_TYPE(__iA)==NPY_INT64);
      ASSERT(PyArray_TYPE(__iB)==NPY_INT64);
      ASSERT(PyArray_TYPE(__iC)==NPY_INT64);

      long nA = PyArray_SIZE(__iA);
      long nB = PyArray_SIZE(__iB);
      long nC = PyArray_SIZE(__iC);

      int64_t* idxA = (int64_t*)PyArray_DATA(__iA);
      int64_t* idxB = (int64_t*)PyArray_DATA(__iB);
      int64_t* idxC = (int64_t*)PyArray_DATA(__iC);
      
      ASSERT(nA == nB && nA == nC);

      Py_buffer* bufA = PyMemoryView_GET_BUFFER(_vA);
      Py_buffer* bufB = PyMemoryView_GET_BUFFER(_vB);
      Py_buffer* bufC = PyMemoryView_GET_BUFFER(_vC);
      
      ASSERT(PyBuffer_IsContiguous(bufA,'C'));
      ASSERT(PyBuffer_IsContiguous(bufB,'C'));
      ASSERT(PyBuffer_IsContiguous(bufC,'C'));

      void* data_A = bufA->buf;
      void* data_B = bufB->buf;
      void* data_C = bufC->buf;
    
      //size_t size_A = bufA->len;
      //size_t size_B = bufB->len;
      //size_t size_C = bufC->len;

      a_data_A[_i] = data_A;
      a_data_B[_i] = data_B;
      a_data_C[_i] = data_C;

      a_idxA[_i] = idxA;
      a_idxB[_i] = idxB;
      a_idxC[_i] = idxC;

      a_n[_i] = nC;
    }

    ASSERT(PyType_Check(_dtype));
    const char* __dtype = ((PyTypeObject*)_dtype)->tp_name;

    if (!strcmp(__dtype,"numpy.complex64")) {
      ((cgpt_blas*)p)->jobs.push_back(new cgpt_gemm_job<ComplexF>(m,n,k,alpha,a_data_A, a_idxA, oA,a_data_B,a_idxB, oB,beta,a_data_C,a_idxC,a_n,precision));
    } else if (!strcmp(__dtype,"numpy.complex128")) {
      ((cgpt_blas*)p)->jobs.push_back(new cgpt_gemm_job<ComplexD>(m,n,k,alpha,a_data_A, a_idxA, oA,a_data_B,a_idxB, oB,beta,a_data_C,a_idxC,a_n,precision));
    } else {
      ERR("Unknown dtype = %s\n", __dtype);
    }
    return PyLong_FromLong(0);
  });

template<typename jobCF, typename jobCD>
PyObject* blas_unary(PyObject* args, bool to_scalar) {
  void* p;
  long n;
  PyObject* vA, *vC, *iA, *iC, *_dtype;
  
  if (!PyArg_ParseTuple(args, "llOOOOO", &p,&n,
			&vA, &iA,
			&vC, &iC,&_dtype)) {
    return NULL;
  }
  
  ASSERT(PyMemoryView_Check(vA) && PyMemoryView_Check(vC));
  ASSERT(PyArray_Check(iA) && PyArray_Check(iC));
  
  PyArrayObject* _iA = (PyArrayObject*)iA;
  PyArrayObject* _iC = (PyArrayObject*)iC;
  
  ASSERT(PyArray_TYPE(_iA)==NPY_INT64);
  ASSERT(PyArray_TYPE(_iC)==NPY_INT64);
  
  long nA = PyArray_SIZE(_iA);
  long nC = PyArray_SIZE(_iC);
  
  int64_t* idxA = (int64_t*)PyArray_DATA(_iA);
  int64_t* idxC = (int64_t*)PyArray_DATA(_iC);
  
  if (to_scalar) {
    ASSERT(nC == 1);
  } else {
    ASSERT(nA == nC);
  }
  
  Py_buffer* bufA = PyMemoryView_GET_BUFFER(vA);
  Py_buffer* bufC = PyMemoryView_GET_BUFFER(vC);
  
  ASSERT(PyBuffer_IsContiguous(bufA,'C'));
  ASSERT(PyBuffer_IsContiguous(bufC,'C'));
  
  void* data_A = bufA->buf;
  void* data_C = bufC->buf;
  
  size_t size_A = bufA->len;
  size_t size_C = bufC->len;
  
  ASSERT(PyType_Check(_dtype));
  const char* __dtype = ((PyTypeObject*)_dtype)->tp_name;
  
  if (!strcmp(__dtype,"numpy.complex64")) {
    ((cgpt_blas*)p)->jobs.push_back(new jobCF(n,data_A, idxA, data_C,idxC,nA));
  } else if (!strcmp(__dtype,"numpy.complex128")) {
    ((cgpt_blas*)p)->jobs.push_back(new jobCD(n,data_A, idxA, data_C,idxC,nA));
  } else {
    ERR("Unknown dtype = %s\n", __dtype);
  }
  return PyLong_FromLong(0);
}

EXPORT(blas_inv,{
    return blas_unary<cgpt_inv_job<ComplexF>, cgpt_inv_job<ComplexD> >(args, false);
  });

EXPORT(blas_det,{
    return blas_unary<cgpt_det_job<ComplexF>, cgpt_det_job<ComplexD> >(args, false);
  });

EXPORT(blas_execute,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    ((cgpt_blas*)p)->execute();
    
    return PyLong_FromLong(0);
  });
