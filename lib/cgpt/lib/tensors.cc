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

#include "lib.h"

PyObject* cgpt_gamma_tensor_mul_internal(std::string& otype, Gamma g, PyArrayObject* src, bool from_left) {

  long ndim = PyArray_NDIM(src);
  std::vector<long> dim;

#define SPIN(Ns)
#define SPIN_COLOR(Ns,Nc)
#define COLOR(Nc)				\
  PER_VECTOR_TYPE(iVSpin4Color ## Nc)		\
    PER_MATRIX_TYPE(iMSpin4Color ## Nc)
  
#define PER_VECTOR_TYPE(T)				\
  if (otype == get_otype(T<ComplexD>())) {		\
    T<ComplexD> o;					\
    cgpt_numpy_data_layout(o,dim);			\
    ASSERT(cgpt_numpy_import(o,src,dim));		\
    if (from_left) {					\
      o = g * o;					\
    } else {						\
      ERR("Right-multiplication not defined");		\
    }							\
    return cgpt_numpy_export(o);			\
  } else						
  
#define PER_MATRIX_TYPE(T)				\
  if (otype == get_otype(T<ComplexD>())) {		\
    T<ComplexD> o;					\
    cgpt_numpy_data_layout(o,dim);			\
    ASSERT(cgpt_numpy_import(o,src,dim));		\
    if (from_left) {					\
      o = g * o;					\
    } else {						\
      o = o * g;					\
    }							\
    return cgpt_numpy_export(o);			\
  } else
  
  #include "spin_color.h"
  PER_VECTOR_TYPE(iVSpin4)
  PER_MATRIX_TYPE(iMSpin4)
  {
    ERR("Unsupported tensor type %s",otype.c_str());
  }

#undef COLOR
#undef SPIN
#undef SPIN_COLOR
#undef PER_TYPE

  return PyLong_FromLong(0);
}

EXPORT(gamma_tensor_mul,{

    PyObject* _src,* _otype;
    long gamma, from_left;
    if (!PyArg_ParseTuple(args, "OOll", &_src, &_otype, &gamma, &from_left)) {
      return NULL;
    }
    
    ASSERT(cgpt_PyArray_Check(_src));

    PyArrayObject* src = (PyArrayObject*)_src;

    ASSERT(gamma >= 0 && gamma < gamma_algebra_map_max);

    Gamma g = gamma_algebra_map[gamma];

    std::string otype;
    cgpt_convert(_otype,otype);

    return cgpt_gamma_tensor_mul_internal(otype,g,src,from_left);
  });
