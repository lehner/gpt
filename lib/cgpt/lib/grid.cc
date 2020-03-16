/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include "lib.h"

EXPORT_BEGIN(create_grid) {
  
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
  
} EXPORT_END();



EXPORT_BEGIN(delete_grid) {

  void* p;
  if (!PyArg_ParseTuple(args, "l", &p)) {
    return NULL;
  }

  ((GridCartesian*)p)->Barrier(); // before a grid goes out of life, we need to synchronize
  delete ((GridCartesian*)p);
  return PyLong_FromLong(0);

} EXPORT_END();



EXPORT_BEGIN(grid_barrier) {

  void* p;
  if (!PyArg_ParseTuple(args, "l", &p)) {
    return NULL;
  }

  ((GridCartesian*)p)->Barrier();
  return PyLong_FromLong(0);

} EXPORT_END();



EXPORT_BEGIN(grid_globalsum) {

  void* p;
  PyObject* o;
  if (!PyArg_ParseTuple(args, "lO", &p,&o)) {
    return NULL;
  }

  GridCartesian* grid = (GridCartesian*)p;
  if (PyComplex_Check(o)) {
    ComplexD c;
    cgpt_convert(o,c);
    grid->GlobalSum(c);
    return PyComplex_FromDoubles(c.real(),c.imag());
  } else if (PyFloat_Check(o)) {
    RealD c;
    cgpt_convert(o,c);
    grid->GlobalSum(c);
    return PyFloat_FromDouble(c);
  } else if (PyLong_Check(o)) {
    uint64_t c;
    cgpt_convert(o,c);
    grid->GlobalSum(c);
    return PyLong_FromLong(c);
  } else if (PyArray_Check(o)) {
    PyArrayObject* ao = (PyArrayObject*)o;
    int dt = PyArray_TYPE(ao);
    void* data = PyArray_DATA(ao);
    size_t nbytes = PyArray_NBYTES(ao);
    if (dt == NPY_FLOAT32 || dt == NPY_COMPLEX64) {
      grid->GlobalSumVector((RealF*)data, nbytes / 4);
    } else if (dt == NPY_FLOAT64 || NPY_COMPLEX128) {
      grid->GlobalSumVector((RealD*)data, nbytes / 8);
    } else {
      std::cerr << "Unsupported numy data type (single, double, csingle, cdouble currently allowed)" << std::endl;
      assert(0);
    }
  } else {
    assert(0);
  }
  // need to act on floats, complex, and numpy arrays PyArrayObject
  //PyArrayObject* p;
  return PyLong_FromLong(0);
} EXPORT_END();
