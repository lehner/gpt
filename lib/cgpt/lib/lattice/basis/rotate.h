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

// need functions that accept list of Field *pointer* and tensor pointers, cannot re-use Grid functions
template<class Field>
void cgpt_basis_rotate(std::vector<Field*> &basis,RealD* Qt,int j0, int j1, int k0,int k1,int Nm) {
  typedef decltype(basis[0]->View()) View;
  auto tmp_v = basis[0]->View();
  std::vector<View> basis_v(basis.size(),tmp_v);
  typedef typename Field::vector_object vobj;
  GridBase* grid = basis[0]->Grid();
      
  for(int k=0;k<basis.size();k++){
    basis_v[k] = basis[k]->View();
  }

  thread_region
  {
    std::vector < vobj > B(Nm); // Thread private
    thread_for_in_region(ss, grid->oSites(),{
	for(int j=j0; j<j1; ++j) B[j]=0.;
      
	for(int j=j0; j<j1; ++j){
	  for(int k=k0; k<k1; ++k){
	    B[j] +=Qt[j*Nm+k] * basis_v[k][ss];
	  }
	}
	for(int j=j0; j<j1; ++j){
	  basis_v[j][ss] = B[j];
	}
      });
  }
}

template<class Field>
void cgpt_linear_combination(Field &result,std::vector<Field*> &basis,RealD* Qt) {
  typedef typename Field::vector_object vobj;
  GridBase* grid = basis[0]->Grid();
  int N = (int)basis.size();
  result.Checkerboard() = basis[0]->Checkerboard();
  auto result_v=result.View();
  thread_for(ss, grid->oSites(),{
      vobj B = Zero();
      for(int k=0; k<N; ++k){
	auto basis_k = basis[k]->View();
	B += Qt[k] * basis_k[ss];
      }
      result_v[ss] = B;
    });
}
