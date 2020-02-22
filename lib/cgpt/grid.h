/*
  CGPT

  Authors: Christoph Lehner 2020
*/
static PyObject* cgpt_create_grid(PyObject* self, PyObject* args) {

  PyObject* _gdimension, * _precision;
  if (!PyArg_ParseTuple(args, "OO", &_gdimension, &_precision)) {
    return NULL;
  }

  std::vector<int> gdimension;
  std::string precision;

  cgpt_convert(_gdimension,gdimension);
  cgpt_convert(_precision,precision);
  
  if (precision == "single") {
  } else if (precision == "double") {
    assert(0);
  } else {
    std::cerr << "Unknown precision: " << precision << std::endl;
    assert(0);
  }

  //printf("Create pointer %p\n",test);
  char* test = new char[12];
  return PyLong_FromVoidPtr(test);
}
