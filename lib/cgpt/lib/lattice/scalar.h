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
void cgpt_lattice_transfer_scalar_device_buffer(Lattice<vobj>& from, void* ptr, long size, long offset, long stride,
						long word_line, long word_stride, bool exp) {

  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  GridBase *Fg = from.Grid();
  ASSERT(!Fg->_isCheckerBoarded);
  int nd = Fg->_ndimension;

  ASSERT( (Fg->lSites() * sizeof(sobj) * stride) <= size );

  Coordinate LocalLatt = Fg->LocalDimensions();
  size_t nsite = 1;
  for(int i=0;i<nd;i++) nsite *= LocalLatt[i];

  Coordinate f_ostride = Fg->_ostride;
  Coordinate f_istride = Fg->_istride;
  Coordinate f_rdimensions = Fg->_rdimensions;

  autoView(from_v,from,exp ? AcceleratorRead : AcceleratorWrite);
  scalar_type* to_v = (scalar_type*)ptr;

  const long words=sizeof(vobj)/sizeof(vector_type);
  if (!word_line) {
    word_line = words;
    word_stride = words;
  }

  if (exp) {
    accelerator_for(idx,nsite,1,{
	
	Coordinate from_coor(nd), base;
	Lexicographic::CoorFromIndex(base,idx,LocalLatt);
	for(int i=0;i<nd;i++)
	  from_coor[i] = base[i];
	
	size_t from_oidx = 0; for(int d=0;d<nd;d++) from_oidx+=f_ostride[d]*(from_coor[d]%f_rdimensions[d]);
	size_t from_lane = 0; for(int d=0;d<nd;d++) from_lane+=f_istride[d]*(from_coor[d]/f_rdimensions[d]);
	
	const vector_type* from = (const vector_type *)&from_v[from_oidx];
	scalar_type* to = &to_v[idx * stride * words + offset * word_line];
	
	scalar_type stmp;
	for(long w=0;w<words;w++){
	  stmp = getlane(from[w], from_lane);
	  to[(w / word_line) * word_stride + w % word_line] = stmp;
	}
      });
  } else {
    accelerator_for(idx,nsite,1,{
	
	Coordinate from_coor(nd), base;
	Lexicographic::CoorFromIndex(base,idx,LocalLatt);
	for(int i=0;i<nd;i++)
	  from_coor[i] = base[i];
	
	size_t from_oidx = 0; for(int d=0;d<nd;d++) from_oidx+=f_ostride[d]*(from_coor[d]%f_rdimensions[d]);
	size_t from_lane = 0; for(int d=0;d<nd;d++) from_lane+=f_istride[d]*(from_coor[d]/f_rdimensions[d]);
	
	vector_type* from = (vector_type *)&from_v[from_oidx]; // reverse from and to language for consistency with above case
	scalar_type* to = &to_v[idx * stride * words + offset * word_line];
	
	scalar_type stmp;
	for(long w=0;w<words;w++){
	  stmp = to[(w / word_line) * word_stride + w % word_line];
	  putlane(from[w], stmp, from_lane);
	}
      });
  }
}
