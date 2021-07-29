/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2020  Daniel Richtmann (daniel.richtmann@ur.de)

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

template<typename vCoeff_t>
cgpt_fermion_operator_base* cgpt_create_coarsenedmatrix(PyObject* args) {

  auto grid_c = get_pointer<GridCartesian>(args,"U_grid");
  auto grid_c_rb = get_pointer<GridRedBlackCartesian>(args,"U_grid_rb");
  int make_hermitian = get_int(args,"make_hermitian");
  int nbasis = get_int(args,"nbasis");
  
#define BASIS_SIZE(n)							\
  if(n == nbasis) {							\
    auto ASelfInv_ptr_list = get_pointer_vec<cgpt_Lattice_base>(args,"U_self_inv"); \
    auto A_ptr_list = get_pointer_vec<cgpt_Lattice_base>(args,"U");     \
    									\
    PVector<Lattice<iMSinglet ##n<vCoeff_t>>> ASelfInv;			\
    PVector<Lattice<iMSinglet ##n<vCoeff_t>>> A;			\
    									\
    cgpt_basis_fill(ASelfInv, ASelfInv_ptr_list);			\
    cgpt_basis_fill(A, A_ptr_list);					\
    									\
    ASSERT(A.size() == 9*ASelfInv.size());				\
    									\
    typedef MultiArgVirtualCoarsenedMatrix<iSinglet<vCoeff_t>, n> CMat; \
    auto cm = new CMat(A, ASelfInv, *grid_c, *grid_c_rb, make_hermitian, 1); \
    return new cgpt_coarse_operator<CMat>(cm);				\
  } else
#include "../basis_size.h"
#undef BASIS_SIZE
    { ERR("Unknown basis size %d", (int)nbasis); }

}
