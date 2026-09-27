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

// norm2(q) != 0 evaluated in the precision of q, exactly as the previous
// norm2(iSinglet) predicate (so |q| below ~1e-23 counts as zero in single
// precision)
template<typename C>
accelerator_inline bool cgpt_where_nonzero(const C& q) {
  auto re = q.real();
  auto im = q.imag();
  return re*re + im*im != 0;
}

// per-lane select, same predicate as before (norm2 != 0); the
// selected value is copied exactly (no arithmetic blending, so non-finite
// values in the branch that is not taken cannot leak)
template<typename C>
accelerator_inline C cgpt_select(const C& q, const C& y, const C& n) {
  return cgpt_where_nonzero(q) ? y : n;
}

template<class S, class V>
accelerator_inline Grid_simd<S,V> cgpt_select(const Grid_simd<S,V>& q, const Grid_simd<S,V>& y, const Grid_simd<S,V>& n) {
  Grid_simd<S,V> r;
  for (int l=0;l<Grid_simd<S,V>::Nsimd();l++)
    r.putlane(cgpt_where_nonzero(q.getlane(l)) ? y.getlane(l) : n.getlane(l), l);
  return r;
}

template<typename S, typename T>
inline void cgpt_where(Lattice<T>& answer, const Lattice<S>& question, const Lattice<T>& yes, const Lattice<T>& no) {

  GridBase* grid = answer.Grid();
  conformable(grid, question.Grid());
  conformable(grid, yes.Grid());
  conformable(grid, no.Grid());

  // question is a singlet of the same SIMD type as the elements of T
  static_assert(GridTypeMapper<S>::count == 1, "where: question must be a singlet");
  static constexpr int n_elements = GridTypeMapper<T>::count;

  autoView(question_v, question, AcceleratorRead);
  autoView(yes_v, yes, AcceleratorRead);
  autoView(no_v, no, AcceleratorRead);
  autoView(answer_v, answer, AcceleratorWriteDiscard);

  accelerator_for(ss, grid->oSites(), grid->Nsimd(), {
      auto q = coalescedReadElement(question_v[ss], 0);
      for (int e=0;e<n_elements;e++) {
	coalescedWriteElement(answer_v[ss],
			      cgpt_select(q,
					  coalescedReadElement(yes_v[ss], e),
					  coalescedReadElement(no_v[ss], e)), e);
      }
    });

}
