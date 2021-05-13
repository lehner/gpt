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

template<class VLattice>
void cgpt_linear_combination(VLattice &result,VLattice &basis,ComplexD* Qt,long n_virtual,long basis_n_block) {

  ASSERT(result.size() % n_virtual == 0);
  ASSERT(basis.size() % n_virtual == 0);
  long n_vec = result.size() / n_virtual;
  long n_basis = basis.size() / n_virtual;
  GridBase* grid = basis[0].Grid();

  for (size_t i=0;i<result.size();i++)
    result[i].Checkerboard() = basis[0].Checkerboard();


#ifndef GRID_HAS_ACCELERATOR

  VECTOR_VIEW_OPEN(result,result_v,CpuWriteDiscard);
  VECTOR_VIEW_OPEN(basis,basis_v,CpuRead);

  typedef typename std::remove_const<decltype(coalescedRead(basis_v[0][0]))>::type vobj;

  thread_for(ss, grid->oSites(),{
  
    for (long j=0;j<n_virtual;j++) {
      for (long i=0;i<n_vec;i++) {
        vobj B = Zero();
	for(long k=0; k<n_basis; k++) {
          B += Qt[k + i*n_basis] * coalescedRead(basis_v[k*n_virtual + j][ss]);
	}
        coalescedWrite(result_v[i*n_virtual + j][ss], B);
      }
    }

  });

  VECTOR_VIEW_CLOSE(basis_v);
  VECTOR_VIEW_CLOSE(result_v);
#else
  Vector<ComplexD> Qt_jv(n_basis*n_vec);
  ComplexD * Qt_j = & Qt_jv[0];
  thread_for(k,n_basis*n_vec,{
      Qt_j[k]=Qt[k];
    });

  VECTOR_VIEW_OPEN(result,result_v,AcceleratorWriteDiscard);
  for (long basis_i0=0;basis_i0<n_basis;basis_i0+=basis_n_block) {
    long basis_i1 = std::min(basis_i0 + basis_n_block,n_basis);
    long basis_block = basis_i1 - basis_i0;

    VECTOR_VIEW_OPEN(basis.slice(basis_i0*n_virtual,basis_i1*n_virtual),basis_v,AcceleratorRead);
    long Nsimd = grid->Nsimd();
    long oSites = grid->oSites();
    accelerator_for(_idx, oSites*n_vec*n_virtual,Nsimd,{
	auto idx = _idx;
	auto ss = idx % oSites; idx /= oSites;
	auto vec_i = idx % n_vec; idx /= n_vec;
	auto virtual_i = idx % n_virtual; idx /= n_virtual;

	decltype(coalescedRead(basis_v[0][ss])) B;

	if (basis_i0 == 0)
	  B = Zero();
	else
	  B = coalescedRead(result_v[vec_i*n_virtual + virtual_i][ss]);
	
	for(long basis_i_rel=0; basis_i_rel<basis_block; basis_i_rel++) {
	  long basis_i_abs = basis_i_rel + basis_i0;
	  B += Qt_j[basis_i_abs + vec_i*n_basis] * coalescedRead(basis_v[basis_i_rel*n_virtual + virtual_i][ss]);
	}
	
	coalescedWrite(result_v[vec_i*n_virtual + virtual_i][ss], B);

      });

    VECTOR_VIEW_CLOSE(basis_v);
  }
  VECTOR_VIEW_CLOSE(result_v);
#endif

}


template<class VLattice>
void cgpt_bilinear_combination(VLattice &result,VLattice &left_basis,VLattice &right_basis,ComplexD* Qt,int32_t* left_indices,int32_t* right_indices,
			       long n_virtual, long n_elements) {

  ASSERT(result.size() % n_virtual == 0);
  ASSERT(right_basis.size() % n_virtual == 0);
  ASSERT(left_basis.size() % n_virtual == 0);
  long n_vec = result.size() / n_virtual;
  GridBase* grid = left_basis[0].Grid();

  for (size_t i=0;i<result.size();i++)
    result[i].Checkerboard() = left_basis[0].Checkerboard();


#ifndef GRID_HAS_ACCELERATOR

  VECTOR_VIEW_OPEN(result,result_v,CpuWriteDiscard);
  VECTOR_VIEW_OPEN(left_basis,left_basis_v,CpuRead);
  VECTOR_VIEW_OPEN(right_basis,right_basis_v,CpuRead);

  typedef typename std::remove_const<decltype(coalescedRead(left_basis_v[0][0]))>::type vobj;

  thread_for(ss, grid->oSites(),{
  
    for (long j=0;j<n_virtual;j++) {
      for (long i=0;i<n_vec;i++) {
        vobj B = Zero();
	for(long k=0; k<n_elements; k++) {
	  int32_t left_index = left_indices[k + i*n_elements];
	  int32_t right_index = right_indices[k + i*n_elements];
          B += Qt[k + i*n_elements] * coalescedRead(left_basis_v[left_index*n_virtual + j][ss]) * coalescedRead(right_basis_v[right_index*n_virtual + j][ss]);
	}
        coalescedWrite(result_v[i*n_virtual + j][ss], B);
      }
    }

  });

  VECTOR_VIEW_CLOSE(left_basis_v);
  VECTOR_VIEW_CLOSE(right_basis_v);
  VECTOR_VIEW_CLOSE(result_v);

#else
  Vector<ComplexD> Qt_jv(n_elements*n_vec);
  Vector<int> left_indices_jv(n_elements*n_vec);
  Vector<int> right_indices_jv(n_elements*n_vec);
  ComplexD * Qt_j = & Qt_jv[0];
  int * left_indices_j = & left_indices_jv[0];
  int * right_indices_j = & right_indices_jv[0];
  thread_for(k,n_elements*n_vec,{
      Qt_j[k]=Qt[k];
      left_indices_j[k]=left_indices[k];
      right_indices_j[k]=right_indices[k];
    });

  VECTOR_VIEW_OPEN(result,result_v,AcceleratorWriteDiscard);
  VECTOR_VIEW_OPEN(left_basis,left_basis_v,AcceleratorRead);
  VECTOR_VIEW_OPEN(right_basis,right_basis_v,AcceleratorRead);

  long Nsimd = grid->Nsimd();
  long oSites = grid->oSites();
  accelerator_for(_idx, oSites*n_vec*n_virtual/**n_elements*/,Nsimd,{
      auto idx = _idx;
      auto ss = idx % oSites; idx /= oSites;
      auto vec_i = idx % n_vec; idx /= n_vec;
      auto virtual_i = idx % n_virtual; idx /= n_virtual;
      //auto k = idx;

      decltype(coalescedRead(left_basis_v[0][ss])) B;

      //if (k == 0) {
      B = Zero();
      //} else {
      //B = result_v[vec_i*n_virtual + virtual_i](ss);
      //}
      
      for(long k=0; k<n_elements; k++) {
	int left_index = left_indices_j[k + vec_i*n_elements];
	int right_index = right_indices_j[k + vec_i*n_elements];
	B += Qt_j[k + vec_i*n_elements] * coalescedRead(left_basis_v[left_index*n_virtual + virtual_i][ss]) *
	  coalescedRead(right_basis_v[right_index*n_virtual + virtual_i][ss]);
      }
      
      coalescedWrite(result_v[vec_i*n_virtual + virtual_i][ss], B);
      
    });

  VECTOR_VIEW_CLOSE(left_basis_v);
  VECTOR_VIEW_CLOSE(right_basis_v);
  VECTOR_VIEW_CLOSE(result_v);
#endif

  
}

template<class VField, class Matrix>
void cgpt_basis_rotate_cpu(VField &basis,Matrix& Qt,int j0, int j1, int k0,int k1,int Nm) {
  typedef decltype(basis[0]) Field;
  typedef decltype(basis[0].View(CpuRead)) View;

  Vector<View> basis_v; basis_v.reserve(basis.size());
  typedef typename std::remove_reference<decltype(basis_v[0][0])>::type vobj;
  typedef typename std::remove_reference<decltype(Qt(0,0))>::type Coeff_t;
  GridBase* grid = basis[0].Grid();
      
  for(int k=0;k<basis.size();k++){
    basis_v.push_back(basis[k].View(CpuWrite));
  }

  int max_threads = thread_max();
  Vector < vobj > Bt(Nm * max_threads);
  thread_region
    {
      vobj* B = &Bt[Nm * thread_num()];
      thread_for_in_region(ss, grid->oSites(),{
	  for(int j=j0; j<j1; ++j) B[j]=0.;
      
	  for(int j=j0; j<j1; ++j){
	    for(int k=k0; k<k1; ++k){
	      B[j] +=Qt(j,k) * basis_v[k][ss];
	    }
	  }
	  for(int j=j0; j<j1; ++j){
	    basis_v[j][ss] = B[j];
	  }
	});
    }

  for(int k=0;k<basis.size();k++) basis_v[k].ViewClose();
}
