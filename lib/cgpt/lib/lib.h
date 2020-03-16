/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION 
#define PY_ARRAY_UNIQUE_SYMBOL cgpt_ARRAY_API
#ifndef _THIS_IS_INIT_
#define NO_IMPORT_ARRAY
#endif
#include <numpy/arrayobject.h>
#include <vector>
#include <string>
#include <iostream>

#include <Grid/Grid.h>

using namespace Grid;

#include "exception.h"
#include "convert.h"
#include "numpy.h"
#include "peekpoke.h"
#include "lattice.h"
#include "transform.h"
#include "util.h"
#include "expression.h"
