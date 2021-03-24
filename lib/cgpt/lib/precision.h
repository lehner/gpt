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
#define PER_TENSOR_TYPE(T)						\
  template<typename vtype>						\
  void cgpt_lattice_convert_from(Lattice< T<vtype> >& dst,cgpt_Lattice_base* src) { \
    if (src->type() == typeid(T<vtype>).name()) {			\
      dst = ((cgpt_Lattice<T<vtype>>*)src)->l;				\
    } else if (src->type() == typeid(T<vComplexD>).name()) {		\
      cgpt_precisionChange(dst, ((cgpt_Lattice<T<vComplexD>>*)src)->l ); \
    } else if (src->type() == typeid(T<vComplexF>).name()) {		\
      cgpt_precisionChange(dst, ((cgpt_Lattice<T<vComplexF>>*)src)->l ); \
    } else {								\
      ERR("Only support conversion between single, double");		\
    }									\
  }
#include "tensors.h"
#undef PER_TENSOR_TYPE
