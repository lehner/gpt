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
#include "kernel.h"

EXPORT(create_kernel,{
    return PyLong_FromVoidPtr((void*)new cgpt_kernel());
  });

EXPORT(delete_kernel,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    delete ((cgpt_kernel*)p);
    return PyLong_FromLong(0);
  });

EXPORT(kernel_accumulate,{
    long n, _zero;
    void* p;
    PyObject* v, *_dtype, *_scales;

    if (!PyArg_ParseTuple(args, "llOOOl", &p, &n, &v, &_dtype, &_scales, &_zero)) {
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

    int Nm;
    
    if (!strcmp(__dtype,"numpy.complex64")) {

      ComplexF* scales;
      cgpt_numpy_import_vector(_scales,scales,Nm);
      ASSERT(Nm == (int)n_buf - 1);

      ((cgpt_kernel*)p)->jobs.push_back(new cgpt_accumulate_job<ComplexF>(n,a_data,scales,_zero));
    } else if (!strcmp(__dtype,"numpy.complex128")) {

      ComplexD* scales;
      cgpt_numpy_import_vector(_scales,scales,Nm);
      ASSERT(Nm == (int)n_buf - 1);
      
      ((cgpt_kernel*)p)->jobs.push_back(new cgpt_accumulate_job<ComplexD>(n,a_data,scales,_zero));
    } else {
      ERR("Unknown dtype = %s\n", __dtype);
    }
    return PyLong_FromLong(0);
  });

EXPORT(kernel_indexed_sum,{
    void* p;
    long id, ts, acc;
    PyObject* sv, *iv, *tv, *_dtype;
    PyObject* ss;

    if (!PyArg_ParseTuple(args, "lOOOlOlOl", &p, &sv, &ss,
			  &iv, &id, &tv, &ts, &_dtype, &acc)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(sv));
    ASSERT(PyMemoryView_Check(tv));
    ASSERT(PyMemoryView_Check(iv));

    Py_buffer* sb = PyMemoryView_GET_BUFFER(sv);
    Py_buffer* tb = PyMemoryView_GET_BUFFER(tv);
    Py_buffer* ib = PyMemoryView_GET_BUFFER(iv);
    ASSERT(PyBuffer_IsContiguous(sb,'C'));
    ASSERT(PyBuffer_IsContiguous(ib,'C'));
    ASSERT(PyBuffer_IsContiguous(tb,'C'));

    void* sp = sb->buf;
    void* tp = tb->buf;
    void* ip = ib->buf;

    ASSERT(PyType_Check(_dtype));
    const char* __dtype = ((PyTypeObject*)_dtype)->tp_name;

    std::vector<long> _ss;
    cgpt_convert(ss, _ss);

    if (!strcmp(__dtype,"numpy.complex64")) {
      ((cgpt_kernel*)p)->jobs.push_back(new cgpt_indexed_sum_job<ComplexF>(sp,tp,ip,_ss,ts,id,acc));
    } else if (!strcmp(__dtype,"numpy.complex128")) {
      ((cgpt_kernel*)p)->jobs.push_back(new cgpt_indexed_sum_job<ComplexD>(sp,tp,ip,_ss,ts,id,acc));
    } else {
      ERR("Unknown dtype = %s\n", __dtype);
    }
    return PyLong_FromLong(0);
  });

EXPORT(kernel_contract,{
    void* p;
    PyObject* _tensors, *_strides, *_dimensions, *_conjugate, *_dtype;

    std::vector<void*> tensors;
    std::vector<std::vector<long>> strides;
    std::vector<long> dimensions;
    std::vector<long> conjugate;

    if (!PyArg_ParseTuple(args, "lOOOOO", &p,
			  &_tensors, &_strides, &_dimensions,
			  &_conjugate, &_dtype)) {
      return NULL;
    }

    ASSERT(PyList_Check(_tensors));
    size_t ntensors = PyList_Size(_tensors);
    tensors.resize(ntensors);
    for (int _i=0;_i<ntensors;_i++) {
      PyObject* v = PyList_GetItem(_tensors,_i);
      ASSERT(PyMemoryView_Check(v));
      Py_buffer* b = PyMemoryView_GET_BUFFER(v);
      ASSERT(PyBuffer_IsContiguous(b,'C'));
      void* p = b->buf;
      tensors[_i] = p;
    }
    
    ASSERT(PyType_Check(_dtype));
    const char* __dtype = ((PyTypeObject*)_dtype)->tp_name;

    cgpt_convert(_strides, strides);
    cgpt_convert(_dimensions, dimensions);
    cgpt_convert(_conjugate, conjugate);

    if (!strcmp(__dtype,"numpy.complex64")) {
      ((cgpt_kernel*)p)->jobs.push_back(new cgpt_contract_job<ComplexF>(tensors, strides, dimensions, conjugate));
    } else if (!strcmp(__dtype,"numpy.complex128")) {
      ((cgpt_kernel*)p)->jobs.push_back(new cgpt_contract_job<ComplexD>(tensors, strides, dimensions, conjugate));
    } else {
      ERR("Unknown dtype = %s\n", __dtype);
    }
    return PyLong_FromLong(0);
  });

EXPORT(kernel_gemm,{
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
      ((cgpt_kernel*)p)->jobs.push_back(new cgpt_gemm_job<ComplexF>(m,n,k,alpha,a_data_A, a_idxA, oA,a_data_B,a_idxB, oB,beta,a_data_C,a_idxC,a_n,precision));
    } else if (!strcmp(__dtype,"numpy.complex128")) {
      ((cgpt_kernel*)p)->jobs.push_back(new cgpt_gemm_job<ComplexD>(m,n,k,alpha,a_data_A, a_idxA, oA,a_data_B,a_idxB, oB,beta,a_data_C,a_idxC,a_n,precision));
    } else {
      ERR("Unknown dtype = %s\n", __dtype);
    }
    return PyLong_FromLong(0);
  });

template<typename jobCF, typename jobCD>
PyObject* kernel_unary(PyObject* args, bool to_scalar) {
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
    ((cgpt_kernel*)p)->jobs.push_back(new jobCF(n,data_A, idxA, data_C,idxC,nA));
  } else if (!strcmp(__dtype,"numpy.complex128")) {
    ((cgpt_kernel*)p)->jobs.push_back(new jobCD(n,data_A, idxA, data_C,idxC,nA));
  } else {
    ERR("Unknown dtype = %s\n", __dtype);
  }
  return PyLong_FromLong(0);
}

EXPORT(kernel_transpose_device_memory_view,{
    
    PyObject* smv, *dmv, *_shape, *_axes;
    void* p;
    if (!PyArg_ParseTuple(args, "lOOOO", &p, &dmv,&smv,&_shape, &_axes)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(smv));
    ASSERT(PyMemoryView_Check(dmv));

    std::vector<long> shape, axes;
    cgpt_convert(_shape, shape);
    cgpt_convert(_axes, axes);

    Py_buffer* dbuf = PyMemoryView_GET_BUFFER(dmv);
    Py_buffer* sbuf = PyMemoryView_GET_BUFFER(smv);
    ASSERT(PyBuffer_IsContiguous(dbuf,'C'));
    ASSERT(PyBuffer_IsContiguous(sbuf,'C'));
    void* data_dmv = dbuf->buf;
    void* data_smv = sbuf->buf;
    ASSERT(dbuf->len == sbuf->len);
    long element_size = (long)sbuf->len;
    for (int i=0;i<shape.size();i++) { ASSERT(element_size % shape[i] == 0); element_size /= shape[i]; }

    switch (element_size) {
    case sizeof(float):
      ((cgpt_kernel*)p)->jobs.push_back(new cgpt_transpose_device_memory_view_job<float>(data_dmv, data_smv, shape, axes)); break;
    case sizeof(double):
      ((cgpt_kernel*)p)->jobs.push_back(new cgpt_transpose_device_memory_view_job<double>(data_dmv, data_smv, shape, axes)); break;
    case sizeof(ComplexD):
      ((cgpt_kernel*)p)->jobs.push_back(new cgpt_transpose_device_memory_view_job<ComplexD>(data_dmv, data_smv, shape, axes)); break;
    default:
      ERR("Unknown element_size = %ld", element_size);
    }

    return PyLong_FromLong(0);
  });

EXPORT(kernel_inv,{
    return kernel_unary<cgpt_inv_job<ComplexF>, cgpt_inv_job<ComplexD> >(args, false);
  });

EXPORT(kernel_det,{
    return kernel_unary<cgpt_det_job<ComplexF>, cgpt_det_job<ComplexD> >(args, false);
  });

EXPORT(kernel_execute,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    ((cgpt_kernel*)p)->execute();
    
    return PyLong_FromLong(0);
  });

EXPORT(kernel_str,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    return PyUnicode_FromString(((cgpt_kernel*)p)->str().c_str());
  });

EXPORT(kernel_fft,{
    void* p;
    long howmany, size, sign;
    PyObject* s, *d, *_dtype;

    if (!PyArg_ParseTuple(args, "lOOOlll", &p, &s, &d, &_dtype, &howmany, &size, &sign)) {
      return NULL;
    }

    ASSERT(PyMemoryView_Check(s));
    ASSERT(PyMemoryView_Check(d));

    Py_buffer* _buf_s = PyMemoryView_GET_BUFFER(s);
    Py_buffer* _buf_d = PyMemoryView_GET_BUFFER(d);
    ASSERT(PyBuffer_IsContiguous(_buf_s,'C'));
    ASSERT(PyBuffer_IsContiguous(_buf_d,'C'));

    void* _s = _buf_s->buf;
    void* _d = _buf_d->buf;
    
    ASSERT(PyType_Check(_dtype));
    const char* __dtype = ((PyTypeObject*)_dtype)->tp_name;

    if (!strcmp(__dtype,"numpy.complex64")) {
      ((cgpt_kernel*)p)->jobs.push_back(new cgpt_fft_job<ComplexF>(_s, _d, howmany, size, sign));
    } else if (!strcmp(__dtype,"numpy.complex128")) {
      ((cgpt_kernel*)p)->jobs.push_back(new cgpt_fft_job<ComplexD>(_s, _d, howmany, size, sign));
    } else {
      ERR("Unknown dtype = %s\n", __dtype);
    }
    return PyLong_FromLong(0);
  });

EXPORT(kernel_copy,{
    long _plan;
    void* p;
    PyObject* _dst,* _src,* _lattice_view_location;
    std::string lattice_view_location;
    
    if (!PyArg_ParseTuple(args, "llOOO", &p, &_plan, &_dst, &_src, &_lattice_view_location)) {
      return NULL;
    }

    cgpt_convert(_lattice_view_location, lattice_view_location);
    memory_type lattice_view_mt = cgpt_memory_type_from_string(lattice_view_location);

    gm_transfer* plan = (gm_transfer*)_plan;

    std::vector<gm_transfer::memory_view> vdst, vsrc;

    std::vector<PyObject*> lattice_views;

    cgpt_copy_add_memory_views(vdst, _dst, lattice_views, lattice_view_mt);
    cgpt_copy_add_memory_views(vsrc, _src, lattice_views, lattice_view_mt);

    ((cgpt_kernel*)p)->jobs.push_back(new cgpt_copy_job(plan, vdst, vsrc));

    for (auto v : lattice_views)
      Py_XDECREF(v);

    return PyLong_FromVoidPtr(0);
  });
