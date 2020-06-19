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


#define GRID_UNOP(name)   name<decltype(eval(0, arg))>
#define GRID_DEF_UNOP(op, name)\
  template <typename T1, typename std::enable_if<is_lattice<T1>::value||is_lattice_expr<T1>::value,T1>::type * = nullptr> \
    inline auto op(const T1 &arg) ->decltype(LatticeUnaryExpression<GRID_UNOP(name),T1>(GRID_UNOP(name)(), arg)) \
  {\
    return     LatticeUnaryExpression<GRID_UNOP(name),T1>(GRID_UNOP(name)(), arg); \
  }


template<typename vobj> 
iSinglet<vobj> ToSinglet(const iScalar<iSinglet<vobj>>& a) {
  return a._internal;
}

template<typename vobj> 
iSinglet<vobj> ToSinglet(const iSinglet<vobj>& a) {
  return a;
}

template<typename vobj,int N> 
  iVector<iSinglet<vobj>,N> ToSinglet(const iVector<iSinglet<vobj>,N>& a) {
  return a;
}

GridUnopClass(UnaryToSinglet, ToSinglet(a));
GRID_DEF_UNOP(ToSinglet, UnaryToSinglet);
