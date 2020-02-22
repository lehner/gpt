/*
  CGPT

  Authors: Christoph Lehner 2020
*/
static PyObject* cgpt_init(PyObject* self, PyObject* args) {

  PyObject* _args;
  if (!PyArg_ParseTuple(args, "O", &_args)) {
    return NULL;
  }

  std::vector<std::string> sargs;
  cgpt_convert(_args,sargs);

  // make cargs
  std::vector<char*> cargs;
  for (auto& a : sargs) {
    cargs.push_back((char*)a.c_str());
  }

  int argc = (int)sargs.size();
  char** argv = &cargs[0];

  return PyLong_FromLong(0);
}
