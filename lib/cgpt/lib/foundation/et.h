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

template<class vtype,int N> accelerator_inline iVector<vtype,N> transpose(const iVector<vtype,N>&r) { return r; }

// define vector * vector -> vector elementwise multiplication
template<class l,class r,int N> accelerator_inline
  auto operator * (const iVector<l,N>& lhs,const iVector<r,N>& rhs) -> iVector<decltype(lhs._internal[0]*rhs._internal[0]),N>
{
  typedef decltype(lhs._internal[0]*rhs._internal[0]) ret_t;
  iVector<ret_t,N> ret;
  for (int i=0;i<N;i++)
    mult(&ret._internal[i],&lhs._internal[i],&rhs._internal[i]);
  return ret;
}
