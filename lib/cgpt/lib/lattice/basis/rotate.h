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

template<class VLattice, typename dtype>
void cgpt_basis_rotate(VLattice &basis,dtype* Qt,int j0, int j1, int k0,int k1,int Nm,bool use_accelerator) {
  PMatrix<dtype> _Qt(Qt,Nm);
  if (use_accelerator) {
    basisRotate(basis,_Qt,j0,j1,k0,k1,Nm);
  } else {
    cgpt_basis_rotate_cpu(basis,_Qt,j0,j1,k0,k1,Nm);
  }
}
