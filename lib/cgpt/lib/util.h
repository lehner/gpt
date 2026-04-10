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

template<typename vtype>
void cgpt_ferm_to_prop(Lattice<iColourVector<vtype>>& ferm, cgpt_Lattice_base* _prop, int s, int c, bool f2p) {
  Lattice<iColourMatrix<vtype>> & prop = compatible<iColourMatrix<vtype>>(_prop)->l;

  ASSERT(s == 0);
  
  if (f2p) {
    for(int i = 0; i < Nc; i++) {
      pokeColour(prop, peekColour(ferm,i), i, c);
    }
  } else {
    for(int i = 0; i < Nc; i++) {
      pokeColour(ferm, peekColour(prop, i,c),i);
    }
  }

}

template<typename T>
void cgpt_ferm_to_prop(Lattice<T>& ferm, cgpt_Lattice_base* _prop, int s, int c, bool f2p) {
  ERR("not supported");
}

template<typename T>
void cgpt_transpose_device_memory_view(void* _d, void* _s, std::vector<long>& _shape, std::vector<long>& _axes) {
  
  T* d = (T*)_d;
  T* s = (T*)_s;
  ASSERT(_shape.size() == _axes.size());
  ASSERT(d != s);

  size_t ndim = _shape.size();
  ASSERT(ndim > 0);

  // device axes
  HostDeviceVector<size_t> axes(ndim), shape(ndim);
  for (int i=0;i<ndim;i++) {
    axes[i]=_axes[i];
    shape[i]=_shape[i];
  }
  auto v_axes = axes.toDevice();
  auto v_shape = shape.toDevice();
  
  // source strides
  HostDeviceVector<size_t> s_strides(ndim);
  s_strides[ndim-1] = 1;
  for (int i = ndim-2; i >= 0; --i)
    s_strides[i] = s_strides[i+1] * _shape[i+1];
  auto v_s_strides = s_strides.toDevice();

  // destination strides
  HostDeviceVector<size_t> d_strides(ndim);
  d_strides[ndim-1] = 1;
  for (int i = ndim-2; i >= 0; --i)
    d_strides[i] = d_strides[i+1] * _shape[_axes[i+1]];
  auto v_d_strides = d_strides.toDevice();

  size_t total = 1;
  for (auto ii : _shape) total *= ii;

  // copy
  accelerator_for(linear, total, 1, {
      size_t dst = 0;      
      for (size_t i = 0; i < ndim; ++i) {
	size_t coor = (linear / v_s_strides[v_axes[i]]) % v_shape[v_axes[i]];
	dst += coor * v_d_strides[i];
      }      
      d[dst] = s[linear];      
    });
}
