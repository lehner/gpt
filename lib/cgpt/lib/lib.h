/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION 
#define PY_ARRAY_UNIQUE_SYMBOL cgpt_ARRAY_API
#ifndef _THIS_IS_INIT_
#define NO_IMPORT_ARRAY
#endif
#include <numpy/arrayobject.h>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <sstream>

#include "pvector.h"
#include "time.h"
#include "exception.h"
#include "allocator.h"
#include "foundation.h"
#include "reduce.h"
#include "sort.h"
#include "micro_kernel.h"
#include "convert.h"
#include "checksums.h"
#include "parameters.h"
#include "numpy.h"
#include "distribute.h"
#include "transform.h"
#include "grid.h"
#include "lattice.h"
#include "util.h"
#include "precision.h"
#include "expression.h"

