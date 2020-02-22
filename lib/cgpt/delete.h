/*
  CGPT

  Authors: Christoph Lehner 2020
*/

static PyObject* cgpt_delete(PyObject* self, PyObject* args) {
  void* p;
  if (!PyArg_ParseTuple(args, "l", &p)) {
    return NULL;
  }

  delete p;
  return PyLong_FromLong(0);
}
