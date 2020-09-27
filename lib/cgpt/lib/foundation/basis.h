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
  thread_region
    {
      typedef typename std::remove_reference<decltype(basis_v[0][0])>::type vobj;
      std::vector<vobj> B(n_vec*n_virtual);

      thread_for_in_region(ss, grid->oSites(),{
	  
	  for (long i=0;i<n_vec*n_virtual;i++)
	    B[i] = Zero();
      
	  for(long k=0; k<n_basis; ++k) {
	    for (long j=0;j<n_virtual;j++) {
	      auto b = basis_v[k*n_virtual + j][ss];
	      for (long i=0;i<n_vec;i++) {
		B[i*n_virtual + j] += Qt[k + i*n_basis] * b;
	      }
	    }
	  }

	  for (long i=0;i<n_vec*n_virtual;i++)
	    result_v[i][ss] = B[i];
	});
    }
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
    accelerator_for(idx, grid->oSites()*n_vec,Nsimd,{
	auto vec_i = idx % n_vec;
	auto ss = idx / n_vec;

	decltype(coalescedRead(basis_v[0][ss])) B;

	for (long virtual_i=0; virtual_i<n_virtual; virtual_i++) {
	  if (basis_i0 == 0)
	    B = Zero();
	  else
	    B = result_v[vec_i*n_virtual + virtual_i](ss);

	  for(long basis_i_rel=0; basis_i_rel<basis_block; basis_i_rel++) {
	    long basis_i_abs = basis_i_rel + basis_i0;
	    B += Qt_j[basis_i_abs + vec_i*n_basis] * basis_v[basis_i_rel*n_virtual + virtual_i](ss);
	  }

	  coalescedWrite(result_v[vec_i*n_virtual + virtual_i][ss], B);
	}
      });

    VECTOR_VIEW_CLOSE(basis_v);
  }
  VECTOR_VIEW_CLOSE(result_v);
#endif

}
