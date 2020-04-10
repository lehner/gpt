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

template<int N>
class TensorPromote {
 public:
  template<typename T> static accelerator_inline auto ToSinglet(const T& arg) -> decltype(TensorPromote<N-1>::ToSinglet(iScalar<T>())) { iScalar<T> sa; sa._internal=arg; return TensorPromote<N-1>::ToSinglet(sa); };
};

template<>
class TensorPromote<0> {
 public:
  template<typename T> static accelerator_inline const T & ToSinglet(const T& arg) { return arg; };
};

#define ConformSinglet(a,b) TensorPromote<b::TensorLevel>::ToSinglet(TensorRemove(a))
