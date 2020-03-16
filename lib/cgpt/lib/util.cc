/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include "lib.h"

EXPORT_BEGIN(util_ferm2prop) {

  void* _ferm,* _prop;
  long spin, color;
  PyObject* _f2p;
  if (!PyArg_ParseTuple(args, "llllO", &_ferm, &_prop, &spin, &color, &_f2p)) {
    return NULL;
  }

  cgpt_Lattice_base* ferm = (cgpt_Lattice_base*)_ferm;
  cgpt_Lattice_base* prop = (cgpt_Lattice_base*)_prop;
  
  bool f2p;
  cgpt_convert(_f2p,f2p);

  ferm->ferm_to_prop(prop,(int)spin,(int)color,f2p);

  return PyLong_FromLong(0);
} EXPORT_END();
