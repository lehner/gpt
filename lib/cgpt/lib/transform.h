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
PyObject* cgpt_lattice_rank_slice(const PVector<Lattice<T>>& basis, int dim) {
  typedef typename Lattice<T>::vector_object vobj;
  typedef typename vobj::scalar_object sobj;

  std::vector<sobj> result;
  cgpt_rank_slice_sum(basis, result, dim);

  int Nsobj  = result.size() / basis.size();  

  return cgpt_numpy_export(result, Nsobj);
}

template<typename T>
PyObject* cgpt_lattice_rank_indexed_sum(const PVector<Lattice<T>>& basis, const Lattice<iSinglet<typename T::vector_type>>& idx, long Nsobj) {
  typedef typename Lattice<T>::vector_object vobj;
  typedef typename vobj::scalar_object sobj;

  int Nbasis = basis.size();
  std::vector<sobj> result(Nbasis * Nsobj);
  cgpt_rank_indexed_sum(basis, idx, result);

  return cgpt_numpy_export(result, Nsobj);
}
