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

static int infer_numpy_type(const ComplexD& v) { return NPY_COMPLEX128; }
static int infer_numpy_type(const ComplexF& v) { return NPY_COMPLEX64; }
static int infer_numpy_type(const std::string & precision) {
  if (precision == "single") {
    return NPY_COMPLEX64;
  } else if (precision == "double") {
    return NPY_COMPLEX128;
  } else {
    ERR("Unknown precision %s",precision.c_str());
  }
}

static void cgpt_numpy_data_layout(const ComplexF& v, std::vector<long>& dim) {}
static void cgpt_numpy_data_layout(const ComplexD& v, std::vector<long>& dim) {}

template<typename sobj, int N>
  void cgpt_numpy_data_layout(const iMatrix<sobj,N>& v, std::vector<long>& dim);

template<typename sobj, int N>
  void cgpt_numpy_data_layout(const iVector<sobj,N>& v, std::vector<long>& dim);

template<typename sobj>
void cgpt_numpy_data_layout(const iScalar<sobj>& v, std::vector<long>& dim) {
  cgpt_numpy_data_layout(v._internal,dim);
}

template<typename sobj, int N>
void cgpt_numpy_data_layout(const iVector<sobj,N>& v, std::vector<long>& dim) {
  dim.push_back(N);
  cgpt_numpy_data_layout(v._internal[0],dim);
}

template<typename sobj, int N>
void cgpt_numpy_data_layout(const iMatrix<sobj,N>& v, std::vector<long>& dim) {
  dim.push_back(N);
  dim.push_back(N);
  cgpt_numpy_data_layout(v._internal[0][0],dim);
}

template<typename sobj>
PyObject* cgpt_numpy_export(const sobj& v) {
  typedef typename sobj::scalar_type t;
  std::vector<long> dim;
  cgpt_numpy_data_layout(v,dim);

  t* c = (t*)&v;

  if (dim.empty()) {
    return PyComplex_FromDoubles(c[0].real(),c[0].imag());
  }

  
  PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNew((int)dim.size(), &dim[0], infer_numpy_type(*c));
  memcpy(PyArray_DATA(arr),c,sizeof(sobj));
  return (PyObject*)arr;
}

template<typename sobj>
bool cgpt_numpy_import(sobj& dst,PyArrayObject* src,std::vector<long>& dim) {
  typedef typename sobj::scalar_type t;

  int nd = PyArray_NDIM(src);
  if (nd != (int)dim.size())
    return false;
  long* tdim = PyArray_DIMS(src);
  int n = 1;
  for (int i=0;i<nd;i++) {
    if (tdim[i] != dim[i])
      return false;
    n *= dim[i];
  }

  t* c = (t*)&dst;

  int dt = PyArray_TYPE(src);
  if (dt == NPY_COMPLEX64) {
    ComplexF* s = (ComplexF*)PyArray_DATA(src);
#pragma omp parallel for
    for (int i=0;i<n;i++)
      c[i] = (t)s[i];
  } else if (dt == NPY_COMPLEX128) {
    ComplexD* s = (ComplexD*)PyArray_DATA(src);
#pragma omp parallel for
    for (int i=0;i<n;i++)
      c[i] = (t)s[i];
  } else {
    ERR("Incompatible numpy type");
  }

  return true;
}

template<typename sobj>
void cgpt_numpy_import(sobj& dst,PyObject* _src) {
  typedef typename sobj::scalar_type t;
  std::vector<long> dim;
  cgpt_numpy_data_layout(dst,dim);

  if (dim.empty()) {
    ComplexD src;
    cgpt_convert(_src,src);
    t* c = (t*)&dst;
    c[0] = src;
  } else {
    ASSERT(PyArray_Check(_src));
    PyArrayObject* src = (PyArrayObject*)_src;
    if (!cgpt_numpy_import(dst,src,dim))
      ERR("Incompatible types");
  }
}

static void cgpt_numpy_import_matrix(PyObject* _Qt, RealD* & data, int & Nm) {
  ASSERT(PyArray_Check(_Qt));
  PyArrayObject* Qt = (PyArrayObject*)_Qt;
  ASSERT(PyArray_NDIM(Qt)==2);
  Nm = PyArray_DIM(Qt,0);
  ASSERT(Nm == PyArray_DIM(Qt,1));
  // TODO: check and at least forbid strides
  ASSERT(PyArray_TYPE(Qt) == NPY_FLOAT64);
  data = (RealD*)PyArray_DATA(Qt);
}

static void cgpt_numpy_import_vector(PyObject* _Qt, RealD* & data, int & Nm) {
  ASSERT(PyArray_Check(_Qt));
  PyArrayObject* Qt = (PyArrayObject*)_Qt;
  ASSERT(PyArray_NDIM(Qt)==1);
  Nm = PyArray_DIM(Qt,0);
  // TODO: check and at least forbid strides
  ASSERT(PyArray_TYPE(Qt) == NPY_FLOAT64);
  data = (RealD*)PyArray_DATA(Qt);
}
