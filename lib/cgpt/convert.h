/*
  CGPT

  Authors: Christoph Lehner 2020
*/
void cgpt_convert(PyObject* in, int& out) {
  assert(PyLong_Check(in));
  out = PyLong_AsLong(in);
}

void cgpt_convert(PyObject* in,  std::string& s) {
  if (PyType_Check(in)) {
    s=((PyTypeObject*)in)->tp_name;
  } else {
    assert(0);
  }
}

template<typename t>
void cgpt_convert(PyObject* in, std::vector<t>& out) {
  assert(PyList_Check(in));
  out.resize(PyList_Size(in));
  for (size_t i = 0; i < out.size(); i++)
    cgpt_convert(PyList_GetItem(in,i),out[i]);
}
