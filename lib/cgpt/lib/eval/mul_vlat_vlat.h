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
			int unary,
			bool ac, ComplexD coef) {

  // need at least one for lhs and rhs
  ASSERT(lhs_vl.size() > 0 && rhs_vl.size() > 0);

  // learn singlet tensor structure
  int lhs_singlet_rank = lhs_vl[0]->singlet_rank();
  int lhs_singlet_dim  = size_to_singlet_dim(lhs_singlet_rank, (int)lhs_vl.size());
  int rhs_singlet_rank = rhs_vl[0]->singlet_rank();
  int rhs_singlet_dim  = size_to_singlet_dim(rhs_singlet_rank, (int)rhs_vl.size());

  // SS -> S
  if (lhs_singlet_rank == 0 && rhs_singlet_rank == 0) {
    dst_vl.resize(1, 0);
    dst_vl[0] = lhs_vl[0]->mul( dst_vl[0], ac, rhs_vl[0], lhs_unary, rhs_unary, unary, coef);
    return;
  }

  // SV -> V
  if (lhs_singlet_rank == 0 && rhs_singlet_rank == 1) {
    dst_vl.resize(rhs_singlet_dim, 0);
    for (int idx=0;idx<rhs_singlet_dim;idx++)
      dst_vl[idx] = lhs_vl[0]->mul( dst_vl[idx], ac, rhs_vl[idx], lhs_unary, rhs_unary, unary, coef);
    return;
  }

  // VS -> V
  if (lhs_singlet_rank == 1 && rhs_singlet_rank == 0) {
    dst_vl.resize(lhs_singlet_dim, 0);
    for (int idx=0;idx<lhs_singlet_dim;idx++)
      dst_vl[idx] = lhs_vl[idx]->mul( dst_vl[idx], ac, rhs_vl[0], lhs_unary, rhs_unary, unary, coef);
    return;
  }

  // VV -> S/M
  if (lhs_singlet_rank == 1 && rhs_singlet_rank == 1) {
    if (lhs_unary == 0 && rhs_unary == (BIT_TRANS|BIT_CONJ)) {
      // outer product -> M
      ASSERT(lhs_singlet_dim == rhs_singlet_dim);
      dst_vl.resize(lhs_singlet_dim*rhs_singlet_dim, 0);
      for (int i=0;i<lhs_singlet_dim;i++) {
	for (int j=0;j<rhs_singlet_dim;j++) {
	  int idx = j*lhs_singlet_dim + i;
	  dst_vl[idx] = lhs_vl[i]->mul( dst_vl[idx], ac, rhs_vl[j], lhs_unary, rhs_unary, unary, coef);
	}
      }
      return;
    } else if (lhs_unary == (BIT_TRANS|BIT_CONJ) && rhs_unary == 0) {
      ERR("Not implemented");
      // inner product -> S
      /*ASSERT(lhs_singlet_dim == rhs_singlet_dim);
      dst_vl.resize(1, 0);
      bool _ac = ac;
      for (int i=0;i<lhs_singlet_dim;i++) {
	for (int j=0;j<rhs_singlet_dim;j++) {
	  dst_vl[0] = lhs_vl[i]->mul( dst_vl[0], _ac, rhs_vl[j], lhs_unary, rhs_unary, unary, coef);
	  _ac = true;
	}
      }
      return;*/
    } else {
      ERR("Invalid combination of two vectors");
    }
  }

  // SM -> M
  if (lhs_singlet_rank == 0 && rhs_singlet_rank == 2) {
    int dim = rhs_singlet_dim;
    dst_vl.resize(dim*dim, 0);
    for (int idx=0;idx<dim*dim;idx++) {
      dst_vl[idx] = lhs_vl[0]->mul(dst_vl[idx], ac, rhs_vl[idx], lhs_unary, rhs_unary, unary, coef);
    }
    return;
  }

  // M X -> Y
  if (lhs_singlet_rank == 2) {

#define matrix_index(i,j,trans) ((trans) ? ((i)*dim + (j)) : ((j) * dim + (i)))
    
    if (rhs_singlet_rank == 0) {    // MS -> M
      
      int dim = lhs_singlet_dim;
      dst_vl.resize(dim*dim, 0);
      for (int idx=0;idx<dim*dim;idx++) {
	dst_vl[idx] = lhs_vl[idx]->mul(dst_vl[idx], ac, rhs_vl[0], lhs_unary, rhs_unary, unary, coef);
      }
      return;
      
    } else if (rhs_singlet_rank == 1) {   // MV -> V
      
      ASSERT(lhs_singlet_dim == rhs_singlet_dim);
      int dim = lhs_singlet_dim;
      
      bool mtrans = (lhs_unary & BIT_TRANS) != 0;
      dst_vl.resize(dim, 0);
      for (int i=0;i<dim;i++) {
	
	// init
	dst_vl[i] = lhs_vl[matrix_index(i,0,mtrans)]->
	  mul( dst_vl[i], ac, rhs_vl[0], lhs_unary, rhs_unary, unary, coef);
	
	for (int j=1;j<dim;j++) {
	  lhs_vl[matrix_index(i,j,mtrans)]->
	    mul( dst_vl[i], true, rhs_vl[j], lhs_unary, rhs_unary, unary, coef);
	}
      }
      return;
      
    } else if (rhs_singlet_rank == 2) {   // M M -> M
      ASSERT(lhs_singlet_dim == rhs_singlet_dim);
      int dim = lhs_singlet_dim;
      bool ltrans = (lhs_unary & BIT_TRANS) != 0;
      bool rtrans = (rhs_unary & BIT_TRANS) != 0;
      bool trace = (unary & BIT_COLORTRACE) != 0;
      dst_vl.resize(trace ? 1 : dim*dim, 0);

      for (int i=0;i<dim;i++) {
	if (trace && i != 0)
	  ac = true;
	for (int j=0;j<dim;j++) {

	  if (trace && i != j)
	    continue;
	  
	  int dst_idx = trace ? 0 : matrix_index(i,j,false);
	  
	  // init
	  dst_vl[dst_idx] = lhs_vl[matrix_index(i,0,ltrans)]->
	    mul( ac ? dst_vl[dst_idx] : 0, ac, rhs_vl[matrix_index(0,j,rtrans)], lhs_unary, rhs_unary, unary, coef);

	  for (int l=1;l<dim;l++) {
	    lhs_vl[matrix_index(i,l,ltrans)]->
	      mul( dst_vl[dst_idx], true, rhs_vl[matrix_index(l,j,rtrans)], lhs_unary, rhs_unary, unary, coef);
	  }
	}
      }
      return;
    }

#undef matrix_index

  }

  ERR("Unknown multiplication of singlet rank %d with %d",lhs_singlet_rank,rhs_singlet_rank);
}
