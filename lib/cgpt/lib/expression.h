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
// factor unary
#define BIT_TRANS 1
#define BIT_CONJ 2
#define NUM_FACTOR_UNARY 4

// term unary
#define BIT_SPINTRACE 1
#define BIT_COLORTRACE 2

namespace Grid {
  // need to supplement this for current Grid
  template<class vtype,int N> accelerator_inline iVector<vtype,N> transpose(const iVector<vtype,N>&r) { return r; }
};

// declaration
template<typename T> cgpt_Lattice_base* cgpt_compatible_linear_combination(Lattice<T>& _compatible,cgpt_Lattice_base* dst,bool ac, std::vector<cgpt_lattice_term>& f, int unary_factor, int unary_expr);

#define PER_TENSOR_TYPE(T) \
  template<typename vtype> cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T<vtype>>& la,int unary_b, cgpt_Lattice_base* b, int unary_expr); \
  template<typename vtype> cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T<vtype>>& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool reverse); \
  template<typename vtype> cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T<vtype>>& la, Gamma::Algebra gamma, int unary_expr, bool reverse);

#include "tensors.h"

#undef PER_TENSOR_TYPE

// convert compatible types to singlet
#include "expression/singlet.h"

// unary
#include "expression/unary.h"
