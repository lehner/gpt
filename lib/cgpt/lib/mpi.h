/*
  CGPT

  Authors: Christoph Lehner 2020
*/
static PyObject* cgpt_global_rank(PyObject* self, PyObject* args) {
  return PyLong_FromLong(CartesianCommunicator::RankWorld());
}

