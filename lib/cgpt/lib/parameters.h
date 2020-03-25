/*
  CGPT

  Authors: Christoph Lehner 2020
*/
static PyObject* get_key(PyObject* dict, const char* key) {
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

static RealD get_float(PyObject* dict, const char* key) {
  PyObject* _val = get_key(dict,key);
  RealD val;
  cgpt_convert(_val,val);
  return val;
}

static bool get_bool(PyObject* dict, const char* key) {
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
    PyObject* _lv = PyList_GetItem(val,i);
    ComplexD lv;
    cgpt_convert(_lv,lv);
    ret[i] = lv;
  }
  return ret;
}

static std::vector<ComplexD> get_complex_vec_gen(PyObject* dict, const char* key) {
  PyObject* val = get_key(dict,key);
  ASSERT(PyList_Check(val));
  int N = (int)PyList_Size(val);

  std::vector<ComplexD> ret(N);
  for (int i=0;i<N;i++) {
    PyObject* _lv = PyList_GetItem(val,i);
    ComplexD lv;
    cgpt_convert(_lv,lv);
    ret[i] = lv;
  }
  return ret;
}
