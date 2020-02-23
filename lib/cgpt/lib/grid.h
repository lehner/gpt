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
  
  int nd = (int)gdimension.size();
  int Nsimd;

  if (precision == "single") {
    Nsimd = vComplexF::Nsimd();
  } else if (precision == "double") {
    Nsimd = vComplexD::Nsimd();
  } else {
    std::cerr << "Unknown precision: " << precision << std::endl;
    assert(0);
  }

  GridCartesian* grid;
  if (nd >= 4) {
    std::vector<int> gdimension4d = gdimension; gdimension4d.resize(4);
    GridCartesian* grid4d = SpaceTimeGrid::makeFourDimGrid(gdimension4d, GridDefaultSimd(4,Nsimd), GridDefaultMpi());
    if (nd == 4) {
      grid = grid4d;
    } else if (nd == 5) {
      grid = SpaceTimeGrid::makeFiveDimGrid(gdimension[5],grid4d);
    } else if (nd > 5) {
      std::cerr << "Unknown dimension " << nd << std::endl;
      assert(0);
    }
  } else {
    std::cerr << "Unknown dimension " << nd << std::endl;
    assert(0);
  }

  return PyLong_FromVoidPtr(grid);
}


static PyObject* cgpt_delete_grid(PyObject* self, PyObject* args) {
  void* p;
  if (!PyArg_ParseTuple(args, "l", &p)) {
    return NULL;
  }

  ((GridCartesian*)p)->Barrier(); // before a grid goes out of life, we need to synchronize
  delete ((GridCartesian*)p);
  return PyLong_FromLong(0);
}

static PyObject* cgpt_grid_barrier(PyObject* self, PyObject* args) {
  void* p;
  if (!PyArg_ParseTuple(args, "l", &p)) {
    return NULL;
  }

  ((GridCartesian*)p)->Barrier();
  return PyLong_FromLong(0);
}

static PyObject* cgpt_grid_globalsum(PyObject* self, PyObject* args) {
  void* p;
  PyObject* o;
  if (!PyArg_ParseTuple(args, "lO", &p,&o)) {
    return NULL;
  }

  GridCartesian* grid = (GridCartesian*)p;
  assert(0); // not yet implemented
  // need to act on floats, complex, and numpy arrays PyArrayObject
  //PyArrayObject* p;
  return PyLong_FromLong(0);
}
