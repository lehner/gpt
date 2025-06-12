/*
    GPT - Grid Python Toolkit
    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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

template<typename vobj>
void cgpt_lattice_transfer_scalar_device_buffer(PVector<Lattice<vobj>>& _from, long _from_n_virtual, long r,
						void* ptr, long size, std::vector<long>& padding, std::vector<long>& offset,
						bool exp, long threads) {

  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  GridBase *Fg = _from[0].Grid();
  ASSERT(!Fg->_isCheckerBoarded);
  int nd = Fg->_ndimension;

  int ps = (int)padding.size();
  int os = (int)offset.size();
  ASSERT(ps == os);

  VECTOR_VIEW_OPEN(_from,_from_v,exp ? AcceleratorRead : AcceleratorWrite); 

  ASSERT(_from.size() % _from_n_virtual == 0);
  size_t n = (size_t)_from_n_virtual;
  size_t n_lat = _from.size() / n;
  size_t nsqrt = 1;
  size_t leading_dim = 1;

  size_t _from_size = _from.size();

  Coordinate LocalLatt = Fg->LocalDimensions();
  size_t nsite = 1, nsite_padded = 1;
  Coordinate to_stride(nd);
  Coordinate to_dimensions(nd);
  Coordinate to_offset(nd);
  
  // padding has ordering t,z,y,x
  for(int i=0;i<nd;i++) {
    nsite *= LocalLatt[i];
    if (i<ps) {
      to_dimensions[i] = LocalLatt[i] + padding[ps-i-1];
      to_offset[i] = offset[ps-i-1];
    } else {
      to_dimensions[i] = LocalLatt[i];
      to_offset[i] = 0;
    }
    to_stride[i] = nsite_padded;
    nsite_padded *= to_dimensions[i];
  }

  ASSERT( (nsite_padded * sizeof(sobj) * _from_size) == size );
      
  Coordinate f_ostride = Fg->_ostride;
  Coordinate f_istride = Fg->_istride;
  Coordinate f_rdimensions = Fg->_rdimensions;
  scalar_type* to_v = (scalar_type*)ptr;

  const long words=sizeof(vobj)/sizeof(vector_type);

  long self_words = words * n;

  size_t a_dim = 1;
  size_t b_dim = n;
  size_t a_stride = 1;
  size_t b_stride = 1;
  size_t c_stride = 1; // word_stride / word_line

  size_t word_line = words;

  if (r == 2) {
    nsqrt = (size_t)sqrt((double)n);
    ASSERT(nsqrt * nsqrt == n);

    a_dim = nsqrt;
    b_dim = nsqrt;

    leading_dim = (size_t)sqrt(self_words); // sqrt(words) * nsqrt
    ASSERT(leading_dim * leading_dim == self_words);

    if (n > 1) {
      a_stride = leading_dim;
      b_stride = 1;
      word_line = leading_dim / nsqrt;
      c_stride = nsqrt;
    } else {
      a_stride = nsqrt;
      b_stride = 1;
    }
  }

  size_t c_dim = words/word_line;

  size_t n_site_lat_a = nsite * n_lat * a_dim;

  auto _stored_nt = acceleratorThreads();
  //_stored_nt * sizeof(vComplexF) / sizeof(ComplexF)
  acceleratorThreads(threads);

  if (exp) {
#ifdef GRID_HAS_ACCELERATOR
    accelerator_for(idx_site_lat_a_lane,n_site_lat_a*word_line,1,{
      size_t idx_site_lat_a = idx_site_lat_a_lane / word_line;
      size_t idx_lane = idx_site_lat_a_lane % word_line;
#else
    accelerator_for(idx_site_lat_a,n_site_lat_a,1,{
#endif
      
      size_t idx_site_lat = idx_site_lat_a / a_dim;
      size_t a = idx_site_lat_a % a_dim;
      
      size_t site = idx_site_lat / n_lat;
      size_t rhs = idx_site_lat % n_lat;
      
      Coordinate from_coor;
      Lexicographic::CoorFromIndex(from_coor,site,LocalLatt);
      
      size_t from_oidx = 0; for(int d=0;d<nd;d++) from_oidx+=f_ostride[d]*(from_coor[d]%f_rdimensions[d]);
      size_t from_lane = 0; for(int d=0;d<nd;d++) from_lane+=f_istride[d]*(from_coor[d]/f_rdimensions[d]);
      size_t to_site = 0; for(int d=0;d<nd;d++) to_site+=to_stride[d]*(from_coor[d] + to_offset[d]);

      for (size_t c=0;c<c_dim;c++) {
	for (size_t b=0;b<b_dim;b++) {

	  size_t idx_inner = (a * c_dim + c) * b_dim + b;
	  size_t line_idx = (to_site * n_lat + rhs) * a_dim * c_dim * b_dim + idx_inner;
	  size_t _i = a + b * a_dim;

	  const vector_type* from = &((const vector_type *)&_from_v[rhs*n+_i][from_oidx])[c * word_line];
	  scalar_type* to = &to_v[line_idx * word_line];
	  
	  scalar_type stmp;
#ifndef GRID_HAS_ACCELERATOR
	  for (long w_fast=0;w_fast<word_line;w_fast++) {
#else
          { auto w_fast = idx_lane;
#endif
	    stmp = getlane(from[w_fast], from_lane);
	    to[w_fast] = stmp;
	  }
	}
      }
    });
  } else {
#ifdef GRID_HAS_ACCELERATOR
    accelerator_for(idx_site_lat_a_lane,n_site_lat_a*word_line,1,{
      size_t idx_site_lat_a = idx_site_lat_a_lane / word_line;
      size_t idx_lane = idx_site_lat_a_lane % word_line;
#else
    accelerator_for(idx_site_lat_a,n_site_lat_a,1,{
#endif

      size_t idx_site_lat = idx_site_lat_a / a_dim;
      size_t a = idx_site_lat_a % a_dim;
      
      size_t site = idx_site_lat / n_lat;
      size_t rhs = idx_site_lat % n_lat;
      
      Coordinate from_coor;
      Lexicographic::CoorFromIndex(from_coor,site,LocalLatt);
      
      size_t from_oidx = 0; for(int d=0;d<nd;d++) from_oidx+=f_ostride[d]*(from_coor[d]%f_rdimensions[d]);
      size_t from_lane = 0; for(int d=0;d<nd;d++) from_lane+=f_istride[d]*(from_coor[d]/f_rdimensions[d]);
      size_t to_site = 0; for(int d=0;d<nd;d++) to_site+=to_stride[d]*(from_coor[d] + to_offset[d]);
      
      for (size_t c=0;c<c_dim;c++) {
	for (size_t b=0;b<b_dim;b++) {

	  size_t idx_inner = (a * c_dim + c) * b_dim + b;
	  size_t line_idx = (to_site * n_lat + rhs) * a_dim * c_dim * b_dim + idx_inner;
	  size_t _i = a + b * a_dim;

	  vector_type* from = &((vector_type *)&_from_v[rhs*n+_i][from_oidx])[c * word_line]; // from/to switch language
	  scalar_type* to = &to_v[line_idx * word_line];
	  
	  scalar_type stmp;
#ifndef GRID_HAS_ACCELERATOR
	  for (long w_fast=0;w_fast<word_line;w_fast++) {
#else
          { auto w_fast = idx_lane;
#endif
	    stmp = to[w_fast];
	    putlane(from[w_fast], stmp, from_lane);
	  }
	}
      }
    });
  }

  acceleratorThreads(_stored_nt);
  VECTOR_VIEW_CLOSE(_from_v);
}
