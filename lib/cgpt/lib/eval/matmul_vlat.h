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
void eval_matmul_vlat(std::vector<cgpt_Lattice_base*> & dst_vl,
		      std::vector<cgpt_Lattice_base*> & lhs_vl,
		      int lhs_unary, 
		      PyArrayObject* rhs_array, 
		      std::vector<std::string> & rhs_v_otype, 
		      int rhs_unary, 
		      int unary,
		      bool rev) {

  ASSERT(lhs_vl.size() > 0);
  ASSERT(dst_vl.size() == 0);

  // learn singlet tensor structure
  int lhs_singlet_rank = lhs_vl[0]->singlet_rank();
  int lhs_singlet_dim  = size_to_singlet_dim(lhs_singlet_rank, (int)lhs_vl.size());
  int rhs_singlet_rank = _otype_singlet_rank_[rhs_v_otype[0]];
  int rhs_singlet_dim  = size_to_singlet_dim(rhs_singlet_rank, (int)rhs_v_otype.size());
  ASSERT(lhs_singlet_dim == rhs_singlet_dim);

  // create temporary block arrays
  std::vector<PyArrayObject*> rhs_v_array(rhs_v_otype.size());
  if (rhs_v_otype.size() == 1) {
    rhs_v_array[0] = rhs_array;
  } else {
    // create array
    ASSERT(rhs_singlet_rank != 0);
    ASSERT(PyArray_NDIM(rhs_array) == rhs_singlet_rank);
    long* rhs_shape = PyArray_DIMS(rhs_array);
    int dtype = PyArray_TYPE(rhs_array);
    long bytes = PyArray_NBYTES(rhs_array);
    char* s = (char*)PyArray_DATA(rhs_array);

    if (rhs_singlet_rank == 1) {

      // V

      ASSERT(rhs_shape[0] % rhs_singlet_dim == 0);
      long rhs_block_size = rhs_shape[0] / rhs_singlet_dim;
      long element_bytes = bytes / rhs_shape[0];
      
      // create vector components
      for (int i=0;i<rhs_singlet_dim;i++) {
	long dim[1] = { rhs_block_size };
	PyArrayObject* a = cgpt_new_PyArray(1, &dim[0], dtype);
	rhs_v_array[i] = a;
	char* d = (char*)PyArray_DATA(a);
	thread_for(idx,rhs_block_size,{
	    memcpy(d + element_bytes*idx,s + element_bytes*(idx + i*rhs_block_size),element_bytes);
	  });
      }

    } else if (rhs_singlet_rank == 2) {

      // M

      ASSERT(rhs_shape[0] % rhs_singlet_dim == 0);
      ASSERT(rhs_shape[1] == rhs_shape[0]);
      ASSERT(rhs_v_array.size() == rhs_singlet_dim * rhs_singlet_dim);
      long rhs_block_size = rhs_shape[0] / rhs_singlet_dim;
      long element_bytes = bytes / rhs_shape[0] / rhs_shape[1];
      
      // create vector components
      for (int i=0;i<rhs_singlet_dim;i++) {
	for (int j=0;j<rhs_singlet_dim;j++) {
	  int vidx = j*rhs_singlet_dim + i;
	  long dim[2] = { rhs_block_size, rhs_block_size };
	  PyArrayObject* a = cgpt_new_PyArray(2, &dim[0], dtype);
	  rhs_v_array[vidx] = a;
	  char* d = (char*)PyArray_DATA(a);
	  thread_for(ii,rhs_block_size,{
	      for (int jj=0;jj<rhs_block_size;jj++) {
		int sidx = (ii + i*rhs_block_size)*rhs_shape[0] + (jj + j*rhs_block_size);
		int didx = ii*rhs_block_size + jj;
		memcpy(d + element_bytes*didx,s + element_bytes*sidx,element_bytes);
	      }
	    });

	}
      }


    } else {
      ERR("Unsupported tensor of rank %d",rhs_singlet_rank);
    }
  }

  if (lhs_singlet_rank == 0 && rhs_singlet_rank == 0) {

    // SS -> S

    dst_vl.resize(1);
    dst_vl[0] = lhs_vl[0]->matmul( 0, false, rhs_v_array[0], rhs_v_otype[0], rhs_unary, lhs_unary, unary, rev);

  } else if (lhs_singlet_rank == 1 && rhs_singlet_rank == 1) {

    // VV -> V

    dst_vl.resize(rhs_singlet_dim);
    for (int idx=0;idx<rhs_singlet_dim;idx++)
      dst_vl[idx] = lhs_vl[idx]->matmul( 0, false, rhs_v_array[idx], rhs_v_otype[idx], rhs_unary, lhs_unary, unary, rev);

  } else if (lhs_singlet_rank == 1 && rhs_singlet_rank == 2 && rev) {

    // MV -> V
    int dim = lhs_singlet_dim;
    bool mtrans = (rhs_unary & BIT_TRANS) != 0;
    dst_vl.resize(dim);

    for (int i=0;i<dim;i++) {

      int idx;
      idx = mtrans ? (i * dim) : (i);

      // init
      dst_vl[i] = lhs_vl[0]->
	matmul( 0, false, rhs_v_array[idx], rhs_v_otype[idx], rhs_unary, lhs_unary, unary, rev);

      for (int j=1;j<dim;j++) {
	idx = mtrans ? (i*dim + j) : (j * dim + i);

	lhs_vl[j]->
	  matmul( dst_vl[i], true, rhs_v_array[idx], rhs_v_otype[idx], rhs_unary, lhs_unary, unary, rev);
      }
    }

  } else {

    ERR("Unknown multiplication of singlet rank %d with %d",lhs_singlet_rank,rhs_singlet_rank);

  }

  // release temporary arrays
  if (rhs_v_array.size() != 1) {
    for (auto&a : rhs_v_array)
      Py_DECREF(a);
  }
}
