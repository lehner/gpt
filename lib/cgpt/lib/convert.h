/*
  CGPT

  Authors: Christoph Lehner 2020
*/
void cgpt_convert(PyObject* in, int& out) {
  assert(PyLong_Check(in));
  out = PyLong_AsLong(in);
}

void cgpt_convert(PyObject* in, ComplexD& out) {
  if (PyLong_Check(in)) {
    out = PyLong_AsLong(in);
  } else if (PyFloat_Check(in)) {
    out = PyFloat_AsDouble(in);
  } else if (PyComplex_Check(in)) {
    out = ComplexD(PyComplex_RealAsDouble(in),
		   PyComplex_ImagAsDouble(in));
  } else {
    assert(0);
  }
}

void cgpt_convert(PyObject* in,  std::string& s) {
  if (PyType_Check(in)) {
    s=((PyTypeObject*)in)->tp_name;
  } else if (PyBytes_Check(in)) {
    s=PyBytes_AsString(in);
  } else if (PyUnicode_Check(in)) {
    PyObject* temp = PyUnicode_AsEncodedString(in, "UTF-8", "strict");
    assert(temp);
    s=PyBytes_AS_STRING(temp);
    Py_DECREF(temp);
  } else {
    assert(0);
  }
}

template<typename t>
void cgpt_convert(PyObject* in, std::vector<t>& out) {
  if (PyList_Check(in)) {
    out.resize(PyList_Size(in));
    for (size_t i = 0; i < out.size(); i++)
      cgpt_convert(PyList_GetItem(in,i),out[i]);
  } else if (PyTuple_Check(in)) {
    out.resize(PyTuple_Size(in));
    for (size_t i = 0; i < out.size(); i++)
      cgpt_convert(PyTuple_GetItem(in,i),out[i]);
  } else {
    assert(0);
  }
}
