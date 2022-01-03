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
void cgpt_scale_per_coordinate(Lattice<T>& dst,Lattice<T>& src,ComplexD* s,int dim) {

  GridBase* grid = dst.Grid();
  conformable(grid, src.Grid());

  dst.Checkerboard() = src.Checkerboard();

  int L = grid->_gdimensions[dim];
    
  autoView(dst_v, dst, AcceleratorWriteDiscard);
  autoView(src_v, src, AcceleratorRead);

  auto dst_p = &dst_v[0];
  auto src_p = &src_v[0];

  Vector<ComplexD> _S(L);
  ComplexD* S = &_S[0];
  thread_for(idx, L, {
      S[idx] = s[idx];
    });

  if (dim == 0 && grid->_simd_layout[0] == 1) {
    accelerator_for(idx, grid->oSites(), T::Nsimd(), {
        int s_idx = idx % L;
        coalescedWrite(dst_p[idx], coalescedRead(src_p[idx]) * S[s_idx]);
      });
  } else {
    ERR("Not implemented yet");
  }
  
}

// sliceSum from Grid but with vector of lattices as input
template<class vobj>
inline void cgpt_slice_sum(const PVector<Lattice<vobj>> &Data,
			   std::vector<typename vobj::scalar_object> &result,
			   int orthogdim)
{
  ///////////////////////////////////////////////////////
  // FIXME precision promoted summation
  // may be important for correlation functions
  // But easily avoided by using double precision fields
  ///////////////////////////////////////////////////////
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;

  GridBase *grid = Data[0].Grid();
  assert(grid!=NULL);

  const int     Nd = grid->_ndimension;
  const int  Nsimd = grid->Nsimd();
  const int Nbasis = Data.size();

  assert(orthogdim >= 0);
  assert(orthogdim < Nd);

  int fd = grid->_fdimensions[orthogdim];
  int ld = grid->_ldimensions[orthogdim];
  int rd = grid->_rdimensions[orthogdim];

  Vector<vobj> lvSum(rd * Nbasis);         // will locally sum vectors first
  Vector<sobj> lsSum(ld * Nbasis, Zero()); // sum across these down to scalars
  result.resize(fd * Nbasis);              // And then global sum to return the same vector to every node

  int      e1 = grid->_slice_nblock[orthogdim];
  int      e2 = grid->_slice_block [orthogdim];
  int  stride = grid->_slice_stride[orthogdim];
  int ostride = grid->_ostride[orthogdim];

  // sum over reduced dimension planes, breaking out orthog dir
  // Parallel over orthog direction
  VECTOR_VIEW_OPEN(Data, Data_v, AcceleratorRead);
  auto lvSum_p = &lvSum[0];
  typedef decltype(coalescedRead(Data_v[0][0])) CalcElem;

  accelerator_for(r, rd * Nbasis, grid->Nsimd(), {
    CalcElem elem = Zero();

    int n_base = r / rd;
    int so = (r % rd) * ostride; // base offset for start of plane
    for(int n = 0; n < e1; n++){
      for(int b = 0; b < e2; b++){
        int ss = so + n * stride + b;
        elem += coalescedRead(Data_v[n_base][ss]);
      }
    }
    coalescedWrite(lvSum_p[r], elem);
  });
  VECTOR_VIEW_CLOSE(Data_v);

  thread_for(n_base, Nbasis, {
    // Sum across simd lanes in the plane, breaking out orthog dir.
    ExtractBuffer<sobj> extracted(Nsimd); // splitting the SIMD
    Coordinate icoor(Nd);

    for(int rt = 0; rt < rd; rt++){
      extract(lvSum[n_base * rd + rt], extracted);
      for(int idx = 0; idx < Nsimd; idx++){
        grid->iCoorFromIindex(icoor, idx);
        int ldx = rt + icoor[orthogdim] * rd;
        lsSum[n_base * ld + ldx] = lsSum[n_base * ld + ldx] + extracted[idx];
      }
    }

    for(int t = 0; t < fd; t++){
      int pt = t / ld; // processor plane
      int lt = t % ld;
      if ( pt == grid->_processor_coor[orthogdim] ) {
        result[n_base * fd + t] = lsSum[n_base * ld + lt];
      } else {
        result[n_base * fd + t] = Zero();
      }
    }
  });
  scalar_type* ptr = (scalar_type *) &result[0];
  int words = fd * sizeof(sobj) / sizeof(scalar_type) * Nbasis;
  grid->GlobalSumVector(ptr, words);
}

template<class vobj>
inline void cgpt_indexed_sum(const PVector<Lattice<vobj>> &Data,
			     const Lattice<iSinglet<vComplexD>> & Index,
			     std::vector<typename vobj::scalar_object> &result)
{
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;

  GridBase *data_grid = Data[0].Grid();
  GridBase* index_grid = Index.Grid();
  
  const int Nbasis = Data.size();
  constexpr int n_elem = GridTypeMapper<vobj>::count;
  
  size_t len = result.size() / Nbasis;
  ASSERT(result.size() % Nbasis == 0);

  ASSERT(data_grid->oSites() * data_grid->Nsimd() == index_grid->oSites() * index_grid->Nsimd());
  
  size_t index_osites_per_block = (index_grid->oSites() + len - 1) / len;

  Vector<sobj> lsSum(index_osites_per_block * len * Nbasis);
  auto lsSum_p = &lsSum[0];
  
  // first zero blocks
  accelerator_for(ss, lsSum.size(), 1, {
      lsSum_p[ss] = Zero();
    });

  GridView gd(data_grid);
  GridView gi(index_grid);

  int ndim = data_grid->Nd();
  ASSERT(ndim == index_grid->Nd());

  Vector<Coordinate> _in_icoor(index_grid->Nsimd());
  Coordinate* in_icoor = &_in_icoor[0];

  for(int lane=0; lane < index_grid->Nsimd(); lane++){
    in_icoor[lane].resize(ndim);
    index_grid->iCoorFromIindex(in_icoor[lane], lane);
  }

  int in_nsimd = index_grid->Nsimd();
  int data_nsimd = data_grid->Nsimd();
  long index_osites = index_grid->oSites();

  autoView(Index_v, Index, AcceleratorRead);
  auto Index_p = &Index_v[0];

  VECTOR_VIEW_OPEN(Data, Data_v, AcceleratorRead);
  
  accelerator_for(ii,index_osites_per_block,1,{

      Coordinate in_ocoor(ndim);
      int lcoor;

      for (long jj=0;jj<len;jj++) {
	long in_oidx = jj*index_osites_per_block + ii;
	if (in_oidx < index_osites) {
      
	  gi.oCoorFromOindex(in_ocoor, in_oidx);

	  for (int in_lane=0;in_lane<in_nsimd;in_lane++) {

	    long index = (long)((ComplexD*)&Index_p[in_oidx])[in_lane].real();
			
	    uint64_t data_lane = 0;
	    uint64_t data_oidx = 0;
	    
	    for(int mu=0;mu<ndim;mu++) {
	      lcoor = in_ocoor[mu] + gi._rdimensions[mu]*in_icoor[in_lane][mu];
	      data_lane += gd._istride[mu] * (lcoor / gd._rdimensions[mu]);
	      data_oidx += gd._ostride[mu] * (lcoor % gd._rdimensions[mu]);
	    }

	    for (int nb=0;nb<Nbasis;nb++) {
	      for (int i=0;i<n_elem;i++) {
		((scalar_type*)&lsSum[(nb * len + index)*index_osites_per_block + ii])[i] +=
		  ((scalar_type*)&Data_v[nb][data_oidx])[i * data_nsimd + data_lane];
	      }
	    }
	  }
	}
      }	
  });

  VECTOR_VIEW_CLOSE(Data_v);

  thread_for(i, result.size(), {
      sobj x = Zero();
      for (size_t j=0;j<index_osites_per_block;j++)
	x = x + lsSum[i*index_osites_per_block + j];
      result[i] = x;
    });
  
  scalar_type* ptr = (scalar_type *) &result[0];
  int words = len * sizeof(sobj) / sizeof(scalar_type) * Nbasis;
  index_grid->GlobalSumVector(ptr, words);
}
