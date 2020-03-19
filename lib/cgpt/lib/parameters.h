/*
  CGPT

  Authors: Christoph Lehner 2020
*/
PyObject* get_key(PyObject* dict, const char* key) {
  ASSERT(PyDict_Check(dict));
  PyObject* val = PyDict_GetItemString(dict,key);
  if (!val)
    ERR("Did not find parameter %s",key);
  return val;
}

template<typename T>
T* get_pointer(PyObject* dict, const char* key) {
  PyObject* val = get_key(dict,key);
  ASSERT(PyLong_Check(val));
  return (T*)PyLong_AsLong(val);
}

template<typename T>
T* get_pointer(PyObject* dict, const char* key, int mu) {
  PyObject* list = get_key(dict,key);
  ASSERT(PyList_Check(list));
  ASSERT(mu >= 0 && mu < PyList_Size(list));
  PyObject* val = PyList_GetItem(list, mu);
  ASSERT(PyLong_Check(val));
  return (T*)PyLong_AsLong(val);
}

RealD get_float(PyObject* dict, const char* key) {
  PyObject* val = get_key(dict,key);
  ASSERT(PyFloat_Check(val));
  return PyFloat_AsDouble(val);
}

bool get_bool(PyObject* dict, const char* key) {
  PyObject* val = get_key(dict,key);
  ASSERT(PyBool_Check(val));
  return (val == Py_True);
}

template<int N>
AcceleratorVector<ComplexD,N> get_complex_vec(PyObject* dict, const char* key) {
  PyObject* val = get_key(dict,key);
  ASSERT(PyList_Check(val) && PyList_Size(val) == N);

  AcceleratorVector<ComplexD,N> ret(N);
  for (int i=0;i<N;i++) {
    PyObject* lv = PyList_GetItem(val,i);
    ASSERT(PyComplex_Check(lv));
    ret[i] = ComplexD(PyComplex_RealAsDouble(lv),PyComplex_ImagAsDouble(lv));
  }
  return ret;
}
