/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Daniel Richtmann (daniel.richtmann@ur.de, https://github.com/lehner/gpt)
                  2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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
#define prec_double 2
#define prec_single 1

template<typename vCoeff_t>
struct FinestLevelCoarseComplex {
  // typedef iSinglet<vCoeff_t> type;
  typedef iScalar<vCoeff_t> type;
};

// this is elegant since we can just add more precisions
template<typename vCoeff_t, int prec = getPrecision<vCoeff_t>::value>
struct FinestLevelFineVec {};
template<typename vCoeff_t>
struct FinestLevelFineVec<vCoeff_t, prec_double> {
  typedef vSpinColourVectorD type;
};
template<typename vCoeff_t>
struct FinestLevelFineVec<vCoeff_t, prec_single> {
  typedef vSpinColourVectorF type;
};

template<typename vCoeff_t>
struct OtherLevelsCoarseComplex {
  typedef iScalar<vCoeff_t> type;
};

template<typename vCoeff_t, int nbasis>
struct OtherLevelsFineVec {
  // typedef iVector<vCoeff_t, nbasis> type;
  typedef iVector<iScalar<vCoeff_t>, nbasis> type;
};

// Rest ///////////////////////////////////////////////////////////////////////

template<typename vCoeff_t>
cgpt_fermion_operator_base* cgpt_create_coarsenedmatrix(PyObject* args) {

  auto grid_c = get_pointer<GridCartesian>(args,"grid_c"); // should actually take an 'F_', and an 'U_' grid
  int hermitian = get_int(args,"hermitian");
  int level = get_int(args,"level"); // 0 = fine, increases with coarser levels
  int nbasis = get_int(args,"nbasis");
  // int nbasis_f = get_int(args,"nbasis_f");
  // int nbasis_c = get_int(args,"nbasis_c");

  // // Tests for the type classes ///////////////////////////////////////////////
  // const int nbasis = 40;
  // char* f0 = typename FinestLevelFineVec<vCoeff_t>::type{};              char* c0 = CoarseComplex{};
  // char* f1 = typename OtherLevelsFineVec<CoarseComplex, nbasis>::type{}; char* c1 = CoarseComplex{};
  // char* f2 = typename OtherLevelsFineVec<CoarseComplex, nbasis>::type{}; char* c2 = CoarseComplex{};

#define BASIS_SIZE(n) \
  if(n == nbasis) { \
    if(level == 0) { \
      typedef CoarsenedMatrix<typename FinestLevelFineVec<vCoeff_t>::type,                          \
                              iSinglet<vCoeff_t>, \
                              n> CMat;                                                       \
      return new cgpt_fermion_operator<CMat>(new CMat(*grid_c, hermitian)); \
    } else {                                                           \
      typedef CoarsenedMatrix<iVSinglet ## n<vCoeff_t>,                \
                              iSinglet<vCoeff_t>,                       \
                              n> CMat; \
      return new cgpt_fermion_operator<CMat>(new CMat(*grid_c, hermitian)); \
      } \
    } else
#include "../basis_size.h"
#undef BASIS_SIZE
  { ERR("Unknown basis size %d", (int)nbasis); }

  // NOTE: With this we should have a default initialized instance of coarsenedmatrix
}

#undef prec_double
#undef prec_single
