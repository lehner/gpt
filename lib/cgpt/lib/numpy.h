/*
  CGPT

  Authors: Christoph Lehner 2020
*/

static int infer_numpy_type(const ComplexD& v) { return NPY_COMPLEX128; }
static int infer_numpy_type(const ComplexF& v) { return NPY_COMPLEX64; }
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
void cgpt_numpy_import(sobj& dst,PyObject* _src) {
  typedef typename sobj::scalar_type t;
  std::vector<long> dim;
  cgpt_numpy_data_layout(dst,dim);

  t* c = (t*)&dst;

  if (dim.empty()) {
    ComplexD src;
    cgpt_convert(_src,src);
    c[0] = src;
  } else {
    ASSERT(PyArray_Check(_src));
    PyArrayObject* src = (PyArrayObject*)_src;
    int nd = PyArray_NDIM(src);
    ASSERT(nd == (int)dim.size());
    long* tdim = PyArray_DIMS(src);
    for (int i=0;i<nd;i++)
      ASSERT(tdim[i] == dim[i]);

    memcpy(c,PyArray_DATA(src),sizeof(sobj));    
  }
}


