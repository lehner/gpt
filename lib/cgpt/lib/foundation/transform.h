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
