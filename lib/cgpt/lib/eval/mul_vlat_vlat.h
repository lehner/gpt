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
static
void eval_mul_vlat_vlat(std::vector<cgpt_Lattice_base*> & dst_vl, 
			std::vector<cgpt_Lattice_base*> & lhs_vl, 
			int lhs_unary, 
			std::vector<cgpt_Lattice_base*> & rhs_vl, 
			int rhs_unary, 
			int unary) {

  // need at least one for lhs and rhs
  ASSERT(lhs_vl.size() > 0 && rhs_vl.size() > 0);

  // learn singlet tensor structure
  int lhs_singlet_rank = lhs_vl[0]->singlet_rank();
  int lhs_singlet_dim  = size_to_singlet_dim(lhs_singlet_rank, (int)lhs_vl.size());
  int rhs_singlet_rank = rhs_vl[0]->singlet_rank();
  int rhs_singlet_dim  = size_to_singlet_dim(rhs_singlet_rank, (int)rhs_vl.size());

  // make sure return is cleared
  ASSERT(dst_vl.size() == 0);

  // SS -> S
  if (lhs_singlet_rank == 0 && rhs_singlet_rank == 0) {
    dst_vl.resize(1);
    dst_vl[0] = lhs_vl[0]->mul( 0, false, rhs_vl[0], lhs_unary, rhs_unary, unary);
    return;
  }

  // SV -> V
  if (lhs_singlet_rank == 0 && rhs_singlet_rank == 1) {
    dst_vl.resize(rhs_singlet_dim);
    for (int idx=0;idx<rhs_singlet_dim;idx++)
      dst_vl[idx] = lhs_vl[0]->mul( 0, false, rhs_vl[idx], lhs_unary, rhs_unary, unary);
    return;
  }

  // VS -> V
  if (lhs_singlet_rank == 1 && rhs_singlet_rank == 0) {
    dst_vl.resize(lhs_singlet_dim);
    for (int idx=0;idx<lhs_singlet_dim;idx++)
      dst_vl[idx] = lhs_vl[idx]->mul( 0, false, rhs_vl[0], lhs_unary, rhs_unary, unary);
    return;
  }

  // SM -> M
  if (lhs_singlet_rank == 0 && rhs_singlet_rank == 2) {
    int dim = rhs_singlet_dim;
    dst_vl.resize(dim*dim);
    for (int idx=0;idx<dim*dim;idx++) {
      dst_vl[idx] = lhs_vl[0]->mul(0, false, rhs_vl[idx], lhs_unary, rhs_unary, unary);
    }
    return;
  }

  // MS -> M
  if (lhs_singlet_rank == 2 && rhs_singlet_rank == 0) {
    int dim = lhs_singlet_dim;
    dst_vl.resize(dim*dim);
    for (int idx=0;idx<dim*dim;idx++) {
      dst_vl[idx] = lhs_vl[idx]->mul(0, false, rhs_vl[0], lhs_unary, rhs_unary, unary);
    }
    return;
  }

  // MV -> V
  if (lhs_singlet_rank == 2 && rhs_singlet_rank == 1) {
    ASSERT(lhs_singlet_dim == rhs_singlet_dim);
    int dim = lhs_singlet_dim;
    bool mtrans = (lhs_unary & BIT_TRANS) != 0;
    dst_vl.resize(dim);
    for (int i=0;i<dim;i++) {

      // init
      dst_vl[i] = lhs_vl[mtrans ? (i * dim) : (i)]->
	mul( 0, false, rhs_vl[0], lhs_unary, rhs_unary, unary);

      for (int j=1;j<dim;j++) {
	lhs_vl[mtrans ? (i*dim + j) : (j * dim + i)]->
	  mul( dst_vl[i], true, rhs_vl[j], lhs_unary, rhs_unary, unary);
      }
    }
    return;
  }

  ERR("Unknown multiplication of singlet rank %d with %d",lhs_singlet_rank,rhs_singlet_rank);
}
