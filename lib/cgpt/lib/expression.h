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
#define BITS_ADJ (BIT_TRANS|BIT_CONJ)
#define NUM_FACTOR_UNARY 4

// term unary
#define BIT_SPINTRACE 1
#define BIT_COLORTRACE 2

// mapping to gamma matrices
static int gamma_algebra_map_max = 12;

static Gamma::Algebra gamma_algebra_map[] = {
  Gamma::Algebra::GammaX, // 0
  Gamma::Algebra::GammaY, // 1
  Gamma::Algebra::GammaZ, // 2
  Gamma::Algebra::GammaT, // 3
  Gamma::Algebra::Gamma5,  // 4
  Gamma::Algebra::SigmaXY, // 5
  Gamma::Algebra::SigmaXZ, // 6
  Gamma::Algebra::SigmaXT, // 7
  Gamma::Algebra::SigmaYZ, // 8
  Gamma::Algebra::SigmaYT, // 9
  Gamma::Algebra::SigmaZT, // 10
  Gamma::Algebra::Identity // 11
};

// declaration
template<typename T> cgpt_Lattice_base* cgpt_compatible_linear_combination(Lattice<T>& _compatible,cgpt_Lattice_base* dst,bool ac, std::vector<cgpt_lattice_term>& f, int unary_factor, int unary_expr);
template<typename T> cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T>& la,int unary_b, cgpt_Lattice_base* b, int unary_expr,ComplexD coef);
template<typename T> cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T>& la, PyArrayObject* b, std::string& bot,
							    int unary_b, int unary_expr, bool reverse, ComplexD coef);
template<typename T> cgpt_Lattice_base* cgpt_lattice_gammamul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T>& la, Gamma::Algebra gamma, int unary_expr, bool reverse, ComplexD coef);

// unary
#include "expression/unary.h"
