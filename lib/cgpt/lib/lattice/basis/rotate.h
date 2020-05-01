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

  //#ifndef GPU_VEC
#if 0
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
#else
  int nrot = j1-j0;


  uint64_t oSites   =grid->oSites();
  uint64_t siteBlock=(grid->oSites()+nrot-1)/nrot; // Maximum 1 additional vector overhead

  //  printf("BasisRotate %d %d nrot %d siteBlock %d\n",j0,j1,nrot,siteBlock);

  Vector <vobj> Bt(siteBlock * nrot); 
  auto Bp=&Bt[0];

  // GPU readable copy of matrix
  Vector<double> Qt_jv(Nm*Nm);
  double *Qt_p = & Qt_jv[0];
  for(int k=k0;k<k1;++k){
    for(int j=j0;j<j1;++j){
      Qt_p[j*Nm+k]=Qt[j*Nm+k];
    }
  }

  // Block the loop to keep storage footprint down
  for(uint64_t s=0;s<oSites;s+=siteBlock){

    // remaining work in this block
    int ssites=MIN(siteBlock,oSites-s);

    // zero out the accumulators
    accelerator_for(ss,siteBlock*nrot,vobj::Nsimd(),{
	auto z=coalescedRead(Bp[ss]);
	z=Zero();
	coalescedWrite(Bp[ss],z);
      });

    accelerator_for(sj,ssites*nrot,vobj::Nsimd(),{
	
	int j =sj%nrot;
	int jj  =j0+j;
	int ss =sj/nrot;
	int sss=ss+s;

	for(int k=k0; k<k1; ++k){
	  auto tmp = coalescedRead(Bp[ss*nrot+j]);
	  coalescedWrite(Bp[ss*nrot+j],tmp+ Qt_p[jj*Nm+k] * coalescedRead(basis_v[k][sss]));
	}
      });

    accelerator_for(sj,ssites*nrot,vobj::Nsimd(),{
	int j =sj%nrot;
	int jj  =j0+j;
	int ss =sj/nrot;
	int sss=ss+s;
	coalescedWrite(basis_v[jj][sss],coalescedRead(Bp[ss*nrot+j]));
      });
  }
#endif
}

template<class Field>
void cgpt_linear_combination(Field &result,std::vector<Field*> &basis,RealD* Qt) {
  typedef typename Field::vector_object vobj;
  GridBase* grid = basis[0]->Grid();

  result.Checkerboard() = basis[0]->Checkerboard();
  auto result_v=result.View();

  int N = (int)basis.size();

  //#ifndef GPU_VEC
#if 0
  thread_for(ss, grid->oSites(),{
      vobj B = Zero();
      for(int k=0; k<N; ++k){
	auto basis_k = basis[k]->View();
	B += Qt[k] * basis_k[ss];
      }
      result_v[ss] = B;
    });
#else
  typedef decltype(basis[0]->View()) View;
  Vector<View> basis_v(N,result_v);
  for(int k=0;k<N;k++){
    basis_v[k] = basis[k]->View();
  }
  Vector<double> Qt_jv(N);
  double * Qt_j = & Qt_jv[0];
  for(int k=0;k<N;++k) Qt_j[k]=Qt[k];
  accelerator_for(ss, grid->oSites(),vobj::Nsimd(),{
      auto B=coalescedRead(basis_v[0][ss]);
      B=Zero();
      for(int k=0; k<N; ++k){
	B +=Qt_j[k] * coalescedRead(basis_v[k][ss]);
      }
      coalescedWrite(result_v[ss], B);
    });
#endif
}
