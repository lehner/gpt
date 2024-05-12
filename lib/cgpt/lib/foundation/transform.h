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
inline void cgpt_rank_slice_sum(const PVector<Lattice<vobj>> &Data,
				std::vector<typename vobj::scalar_object> &result,
				int orthogdim)
{
  ///////////////////////////////////////////////////////
  // FIXME precision promoted summation
  // may be important for correlation functions
  // But easily avoided by using double precision fields
  ///////////////////////////////////////////////////////
  typedef typename vobj::scalar_object sobj;

  GridBase *grid = Data[0].Grid();
  assert(grid!=NULL);

  const int     Nd = grid->_ndimension;
  const int  Nsimd = grid->Nsimd();
  const int Nbasis = Data.size();

  assert(orthogdim >= 0);
  assert(orthogdim < Nd);

  size_t eblock = 1;
  size_t min_block = 64; // TODO: make this configurable
    
  size_t fd = grid->_fdimensions[orthogdim];
  size_t ld = grid->_ldimensions[orthogdim];
  size_t rd = grid->_rdimensions[orthogdim];

  while (eblock * rd < min_block)
    eblock *= 2;

  Vector<vobj> lvSum(rd * eblock * Nbasis);         // will locally sum vectors first
  Vector<sobj> lsSum(ld * Nbasis, Zero()); // sum across these down to scalars
  result.resize(fd * Nbasis);              // And then global sum to return the same vector to every node

  size_t   e1 = grid->_slice_nblock[orthogdim];
  size_t   e2 = grid->_slice_block [orthogdim];
  size_t   e  = (e1 * e2 + eblock - 1) / eblock;
  size_t   stride = grid->_slice_stride[orthogdim];
  size_t   ostride = grid->_ostride[orthogdim];

  // sum over reduced dimension planes, breaking out orthog dir
  // Parallel over orthog direction
  VECTOR_VIEW_OPEN(Data, Data_v, AcceleratorRead);
  auto lvSum_p = &lvSum[0];
  typedef decltype(coalescedRead(Data_v[0][0])) CalcElem;

  accelerator_for(rr, rd * eblock * Nbasis, (size_t)grid->Nsimd(), {
    CalcElem elem = Zero();

    size_t eb = rr % eblock;
    size_t r = rr / eblock;
    size_t n_base = r / rd;
    size_t so = (r % rd) * ostride; // base offset for start of plane
    for(size_t i = 0; i < e; i++) {
      size_t j = eb + i * eblock;
      if (j < e1*e2) {
	size_t n = j / e2;
	size_t b = j % e2;
	size_t ss = so + n * stride + b;
	elem += coalescedRead(Data_v[n_base][ss]);
      }
    }
    coalescedWrite(lvSum_p[rr], elem);
  });
  VECTOR_VIEW_CLOSE(Data_v);

  thread_for(n_base, Nbasis, {
    // Sum across simd lanes in the plane, breaking out orthog dir.
    ExtractBuffer<sobj> extracted(Nsimd); // splitting the SIMD
    Coordinate icoor(Nd);

    for (size_t eb=0;eb<eblock;eb++) {
      for(size_t rt = 0; rt < rd; rt++) {
	extract(lvSum[(n_base * rd + rt) * eblock + eb], extracted);
	for(size_t idx = 0; idx < Nsimd; idx++){
	  grid->iCoorFromIindex(icoor, idx);
	  size_t ldx = rt + icoor[orthogdim] * rd;
	  lsSum[n_base * ld + ldx] = lsSum[n_base * ld + ldx] + extracted[idx];
	}
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
}

template<class vobj>
inline void cgpt_rank_indexed_sum(const PVector<Lattice<vobj>> &Data,
				  const Lattice<iSinglet<typename vobj::vector_type>> & Index,
				  std::vector<typename vobj::scalar_object> &result)
{
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;

  GridBase *grid = Data[0].Grid();
  ASSERT(grid == Index.Grid());
  
  const int Nbasis = Data.size();
  constexpr int n_elem = GridTypeMapper<vobj>::count;
  
  size_t len = result.size() / Nbasis;
  ASSERT(result.size() % Nbasis == 0);

  size_t index_osites_per_block = (grid->oSites() + len - 1) / len;

  Vector<sobj> lsSum(index_osites_per_block * len * Nbasis);
  auto lsSum_p = &lsSum[0];
  
  // first zero blocks
  accelerator_for(ss, lsSum.size(), 1, {
      lsSum_p[ss] = Zero();
    });

  int Nsimd = grid->Nsimd();
  int ndim = grid->Nd();

  long index_osites = grid->oSites();

  Timer("ris: view");
  
  autoView(Index_v, Index, AcceleratorRead);
  auto Index_p = &Index_v[0];

  VECTOR_VIEW_OPEN(Data, Data_v, AcceleratorRead);

  Timer("ris: accelerator reduction");
  accelerator_for(_idx,index_osites_per_block * n_elem * Nbasis,1,{

      uint64_t idx = _idx;
      uint64_t ii = idx % index_osites_per_block; idx /= index_osites_per_block;
      uint64_t i = idx % n_elem; idx /= n_elem;
      uint64_t nb = idx % Nbasis; idx /= Nbasis;
      
      for (long jj=0;jj<len;jj++) {
	long oidx = jj*index_osites_per_block + ii;
	if (oidx < index_osites) {
      
	  for (int lane=0;lane<Nsimd;lane++) {
	    long index = (long)((scalar_type*)&Index_p[oidx])[lane].real();
			
	    ((scalar_type*)&lsSum_p[(nb * len + index)*index_osites_per_block + ii])[i] +=
	      ((scalar_type*)&Data_v[nb][oidx])[i * Nsimd + lane];
	  }
	}
      }	
  });

  Timer("ris: view");
  VECTOR_VIEW_CLOSE(Data_v);

  Timer("ris: thread reduction");
  thread_for(i, result.size(), {
      sobj x = Zero();
      for (size_t j=0;j<index_osites_per_block;j++)
	x = x + lsSum_p[i*index_osites_per_block + j];
      result[i] = x;
    });

  Timer();
}
