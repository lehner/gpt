/*
  CGPT

  Authors: Christoph Lehner 2020
*/
static void cgpt_convert(PyObject* in, int& out) {
  ASSERT(PyLong_Check(in));
  out = PyLong_AsLong(in);
}

static void cgpt_convert(PyObject* in, bool& out) {
  ASSERT(PyBool_Check(in));
  out = in == Py_True;
}

static void cgpt_convert(PyObject* in, ComplexD& out) {
  if (PyLong_Check(in)) {
    out = PyLong_AsLong(in);
  } else if (PyFloat_Check(in)) {
    out = PyFloat_AsDouble(in);
  } else if (PyComplex_Check(in)) {
    out = ComplexD(PyComplex_RealAsDouble(in),
		   PyComplex_ImagAsDouble(in));
  } else {
    ASSERT(0);
  }
}

static void cgpt_convert(PyObject* in, RealD& out) {
  if (PyLong_Check(in)) {
    out = PyLong_AsLong(in);
  } else if (PyFloat_Check(in)) {
    out = PyFloat_AsDouble(in);
  } else {
    ASSERT(0);
  }
}

static void cgpt_convert(PyObject* in, uint64_t& out) {
  if (PyLong_Check(in)) {
    out = PyLong_AsLong(in);
  } else {
    ASSERT(0);
  }
}

static void cgpt_convert(PyObject* in,  std::string& s) {
  if (PyType_Check(in)) {
    s=((PyTypeObject*)in)->tp_name;
  } else if (PyBytes_Check(in)) {
    s=PyBytes_AsString(in);
  } else if (PyUnicode_Check(in)) {
    PyObject* temp = PyUnicode_AsEncodedString(in, "UTF-8", "strict");
    ASSERT(temp);
    s=PyBytes_AS_STRING(temp);
    Py_DECREF(temp);
  } else {
    ASSERT(0);
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
    ASSERT(0);
  }
}
