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

template<typename T>     struct isSpuriousGridScalar                          : public std::false_type { static constexpr bool notvalue = true; };
template<class T>        struct isSpuriousGridScalar<iScalar<iSinglet<T>>>    : public std::true_type  { static constexpr bool notvalue = false; };

// Remove spurious iScalar
template<typename vobj> 
accelerator_inline
iSinglet<vobj> TensorToSinglet(const iScalar<iSinglet<vobj>>& a) {
  return a._internal;
}

template<typename T1, typename std::enable_if<!isSpuriousGridScalar<T1>::value, T1>::type* = nullptr> 
accelerator_inline
T1 TensorToSinglet(const T1& a) {
  return a;
}

// Make scalar
template<typename T>
accelerator_inline
iScalar<T> TensorMakeScalar(const T& a) {
  iScalar<T> ret;
  ret._internal = a;
  return ret;
}

// Define unary operator to work in ET
#define GRID_UNOP(name)   name<decltype(eval(0, arg))>
#define GRID_DEF_UNOP(op, name)\
  template <typename T1, typename std::enable_if<is_lattice<T1>::value||is_lattice_expr<T1>::value,T1>::type * = nullptr> \
    inline auto op(const T1 &arg) ->decltype(LatticeUnaryExpression<GRID_UNOP(name),T1>(GRID_UNOP(name)(), arg)) \
  {\
    return     LatticeUnaryExpression<GRID_UNOP(name),T1>(GRID_UNOP(name)(), arg); \
						     }

GridUnopClass(UnaryToSinglet, TensorToSinglet(a));
GRID_DEF_UNOP(ToSinglet, UnaryToSinglet);

GridUnopClass(UnaryMakeScalar, TensorMakeScalar(a));
GRID_DEF_UNOP(MakeScalar, UnaryMakeScalar);
