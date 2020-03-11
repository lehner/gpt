/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include "lattice/types.h"
#include "lattice/base.h"
#include "lattice/term.h"
#include "lattice/implementation.h"

EXPORT_BEGIN(create_lattice) {

  void* _grid;
  PyObject* _otype, * _prec;
  if (!PyArg_ParseTuple(args, "lOO", &_grid, &_otype, &_prec)) {
    return NULL;
  }

  GridCartesian* grid = (GridCartesian*)_grid;
  std::string otype;
  std::string prec;

  cgpt_convert(_otype,otype);
  cgpt_convert(_prec,prec);

  void* plat = 0;
  if (otype == "ot_complex") {
    if (prec == "single") {
      plat = new cgpt_Lattice<vTComplexF>(grid);
    } else if (prec == "double") {
      plat = new cgpt_Lattice<vTComplexD>(grid);
    }
  } else if (otype == "ot_mcolor") {
    if (prec == "single") {
      plat = new cgpt_Lattice<vColourMatrixF>(grid);
    } else if (prec == "double") {
      plat = new cgpt_Lattice<vColourMatrixD>(grid);
    }
  }

  if (!plat) {
    std::cerr << "Unknown field type: " << otype << "," << prec << std::endl;  
    ASSERT(0);
  }

  return PyLong_FromVoidPtr(plat);
} EXPORT_END();

EXPORT_BEGIN(delete_lattice) {
  void* p;
  if (!PyArg_ParseTuple(args, "l", &p)) {
    return NULL;
  }

  delete ((cgpt_Lattice_base*)p);
  return PyLong_FromLong(0);
} EXPORT_END();

EXPORT_BEGIN(lattice_set_val) {
  void* p;
  PyObject* _coor,* _val;
  if (!PyArg_ParseTuple(args, "lOO", &p, &_coor,&_val)) {
    return NULL;
  }

  std::vector<int> coor;
  cgpt_convert(_coor,coor);

  ComplexD val;
  cgpt_convert(_val,val);

  cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
  l->set_val(coor,val);

  return PyLong_FromLong(0);
} EXPORT_END();

EXPORT_BEGIN(lattice_to_str) {
  void* p;
  if (!PyArg_ParseTuple(args, "l", &p)) {
    return NULL;
  }

  cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
  return l->to_str();
} EXPORT_END();
