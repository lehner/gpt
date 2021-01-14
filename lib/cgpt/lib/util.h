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
template<typename vtype>
void cgpt_ferm_to_prop(Lattice<iSpinColourVector<vtype>>& ferm, cgpt_Lattice_base* _prop, int s, int c, bool f2p) {
  Lattice<iSpinColourMatrix<vtype>> & prop = compatible<iSpinColourMatrix<vtype>>(_prop)->l;

  if (f2p) {
    for(int j = 0; j < Ns; j++) {
      auto pjs = peekSpin(prop, j, s);
      auto fj  = peekSpin(ferm, j);
      for(int i = 0; i < Nc; i++) {
	pokeColour(pjs, peekColour(fj,i), i, c);
      }
      pokeSpin(prop, pjs, j, s);
    }
  } else {
    for(int j = 0; j < Ns; j++) {
      auto pjs = peekSpin(prop, j, s);
      auto fj  = peekSpin(ferm, j);
      for(int i = 0; i < Nc; i++) {
	pokeColour(fj, peekColour(pjs, i,c),i);
      }
      pokeSpin(ferm, fj, j);
    }
  }

}

template<typename T>
void cgpt_ferm_to_prop(Lattice<T>& ferm, cgpt_Lattice_base* _prop, int s, int c, bool f2p) {
  ERR("not supported");
}
