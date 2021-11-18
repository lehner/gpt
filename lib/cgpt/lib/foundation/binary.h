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
    51 Franklin Street, Fifth Floor, Boston, MA 021basis_virtual_size-1n_virtual_red01 USA.
*/

template<typename T>
inline void cgpt_lower_than(Lattice<T>& answer, const Lattice<T>& a, const Lattice<T>& b) {
  ERR("Not implemented");
}

template<typename vtype>
inline void cgpt_lower_than(Lattice<iSinglet<vtype>>& answer, const Lattice<iSinglet<vtype>>& a, const Lattice<iSinglet<vtype>>& b) {
  GridBase* grid = answer.Grid();
  conformable(grid, a.Grid());
  conformable(grid, b.Grid());

  answer.Checkerboard() = a.Checkerboard();

  typedef typename Lattice<iSinglet<vtype>>::scalar_type stype;
  
  autoView(answer_v, answer, CpuWriteDiscard);
  autoView(a_v, a, CpuRead);
  autoView(b_v, b, CpuRead);

  auto oSites = grid->oSites();
  auto Nsimd = grid->Nsimd();
  thread_for(i, oSites, {
      stype* _answer = (stype*)&answer_v[i];
      stype* _a = (stype*)&a_v[i];
      stype* _b = (stype*)&b_v[i];
      for (int j=0;j<Nsimd;j++) {
	_answer[j] = (_a[j].real() < _b[j].real()) ? 1.0 : 0.0;
      }
    });
}

template<typename T>
inline void cgpt_component_wise_multiply(Lattice<T>& answer, const Lattice<T>& a, const Lattice<T>& b) {
  GridBase* grid = answer.Grid();
  conformable(grid, a.Grid());
  conformable(grid, b.Grid());

  answer.Checkerboard() = a.Checkerboard();

  typedef typename Lattice<T>::vector_type vtype;

  autoView(answer_v, answer, AcceleratorWriteDiscard);
  autoView(a_v, a, AcceleratorRead);
  autoView(b_v, b, AcceleratorRead);
  auto answer_p = (vtype*)&answer_v[0];
  auto a_p = (vtype*)&a_v[0];
  auto b_p = (vtype*)&b_v[0];

  accelerator_for(ss, grid->oSites() * sizeof(T) / sizeof(vtype), grid->Nsimd(), {
      coalescedWrite(answer_p[ss], coalescedRead(a_p[ss]) * coalescedRead(b_p[ss]));
    });
}
