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

template<typename T>
void cgpt_block_project(cgpt_Lattice_base* _coarse, Lattice<T>& fine, std::vector<cgpt_Lattice_base*>& _basis) {

  typedef typename Lattice<T>::vector_type vCoeff_t;

  PVector<Lattice<T>> basis;
  cgpt_basis_fill(basis,_basis);

#define BASIS_SIZE(n) if (n == basis.size()) { blockProject(compatible< iComplexV ## n<vCoeff_t> >(_coarse)->l,fine,basis); } else
#include "basis_size.h"
#undef BASIS_SIZE
  { ERR("Unknown basis size %d",(int)basis.size()); }

}



template<typename T>
void cgpt_block_promote(cgpt_Lattice_base* _coarse, Lattice<T>& fine, std::vector<cgpt_Lattice_base*>& _basis) {

  typedef typename Lattice<T>::vector_type vCoeff_t;

  PVector<Lattice<T>> basis;
  cgpt_basis_fill(basis,_basis);

#define BASIS_SIZE(n) if (n == basis.size()) { blockPromote(compatible< iComplexV ## n<vCoeff_t> >(_coarse)->l,fine,basis); } else
#include "basis_size.h"
#undef BASIS_SIZE
  { ERR("Unknown basis size %d",(int)basis.size()); }

}

template<typename T>
void cgpt_block_orthonormalize(cgpt_Lattice_base* _coarse, Lattice<T>& fine, std::vector<cgpt_Lattice_base*>& _basis) { // fine argument just to automatically detect type

  typedef typename Lattice<T>::vector_type vCoeff_t;

  PVector<Lattice<T>> basis;
  cgpt_basis_fill(basis,_basis);

  blockOrthonormalize(compatible< iSinglet<vCoeff_t> >(_coarse)->l,basis);
}

