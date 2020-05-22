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
void cgpt_basis_rotate(VLattice &basis,RealD* Qt,int j0, int j1, int k0,int k1,int Nm) {
  PMatrix<RealD> _Qt(Qt,Nm);
  basisRotate(basis,_Qt,j0,j1,k0,k1,Nm);
}

template<class Field,class VLattice>
void cgpt_linear_combination(Field &result,VLattice &basis,RealD* Qt) {
  typedef typename Field::vector_object vobj;
  GridBase* grid = basis[0].Grid();

  // TODO: map to basisRotateJ
  result.Checkerboard() = basis[0].Checkerboard();
  auto result_v=result.AcceleratorView(ViewWrite);

  int N = (int)basis.size();

#ifndef GRID_NVCC
  thread_for(ss, grid->oSites(),{
      vobj B = Zero();
      for(int k=0; k<N; ++k){
	auto basis_k = basis[k].View();
	B += Qt[k] * basis_k[ss];
      }
      result_v[ss] = B;
    });
#else
  typedef decltype(basis[0].View()) View;
  Vector<View> basis_v(N,result_v);
  for(int k=0;k<N;k++){
    basis_v[k] = basis[k].AcceleratorView(ViewRead);
  }
  Vector<double> Qt_jv(N);
  double * Qt_j = & Qt_jv[0];
  for(int k=0;k<N;++k) Qt_j[k]=Qt[k];
  accelerator_for(ss, grid->oSites(),vobj::Nsimd(),{
      decltype(coalescedRead(basis_v[0][ss])) B;
      B=Zero();
      for(int k=0; k<N; ++k){
	B +=Qt_j[k] * coalescedRead(basis_v[k][ss]);
      }
      coalescedWrite(result_v[ss], B);
    });
#endif
}
