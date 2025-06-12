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
#define GRID_UNOP(name)   name
#define GRID_DEF_UNOP(op, name)						\
  template <typename T1, typename std::enable_if<is_lattice<T1>::value||is_lattice_expr<T1>::value,T1>::type * = nullptr> \
    inline auto op(const T1 &arg) ->decltype(LatticeUnaryExpression<GRID_UNOP(name),T1>(GRID_UNOP(name)(), arg)) \
  {									\
    return     LatticeUnaryExpression<GRID_UNOP(name),T1>(GRID_UNOP(name)(), arg); \
  }

GridUnopClass(UnaryToSinglet, TensorToSinglet(a));
GRID_DEF_UNOP(ToSinglet, UnaryToSinglet);

GridUnopClass(UnaryMakeScalar, TensorMakeScalar(a));
GRID_DEF_UNOP(MakeScalar, UnaryMakeScalar);




#define DEFINE_SIMPLE_UNOP(name)					\
  template <class scalar>						\
  struct cgpt_ ## name ## _functor {					\
    accelerator scalar operator()(const scalar &a) const { return name(a); } \
  };									\
  static accelerator_inline ComplexD _cgpt_ ## name(const ComplexD & z) { return name(z); }; \
  static accelerator_inline ComplexF _cgpt_ ## name(const ComplexF & z) { return name(z); }; \
  template <class S, class V>						\
  accelerator_inline Grid_simd<S, V> _cgpt_ ## name (const Grid_simd<S, V> &r) { \
    return SimdApply(cgpt_ ## name ## _functor<S>(), r);		\
  }									\
  template<class obj> Lattice<obj> cgpt_ ## name(const Lattice<obj> &rhs_i){ \
    Lattice<obj> ret_i(rhs_i.Grid());					\
    autoView( rhs, rhs_i, AcceleratorRead);				\
    autoView( ret, ret_i, AcceleratorWrite);				\
    ret.Checkerboard() = rhs.Checkerboard();				\
    static constexpr int n_elements = GridTypeMapper<obj>::count;	\
    accelerator_for(ss,rhs_i.Grid()->oSites(),obj::Nsimd(),{		\
	for (int e=0;e<n_elements;e++) {				\
	  auto x = coalescedReadElement(rhs[ss], e);			\
	  coalescedWriteElement(ret[ss], _cgpt_ ## name(x), e);		\
	}								\
      });								\
    return ret_i;							\
  }

DEFINE_SIMPLE_UNOP(abs);
DEFINE_SIMPLE_UNOP(sqrt);
DEFINE_SIMPLE_UNOP(log);
DEFINE_SIMPLE_UNOP(sin);
DEFINE_SIMPLE_UNOP(asin);
DEFINE_SIMPLE_UNOP(cos);
DEFINE_SIMPLE_UNOP(acos);
DEFINE_SIMPLE_UNOP(tan);
DEFINE_SIMPLE_UNOP(atan);
DEFINE_SIMPLE_UNOP(sinh);
DEFINE_SIMPLE_UNOP(asinh);
DEFINE_SIMPLE_UNOP(cosh);
DEFINE_SIMPLE_UNOP(acosh);
DEFINE_SIMPLE_UNOP(tanh);
DEFINE_SIMPLE_UNOP(atanh);
DEFINE_SIMPLE_UNOP(real);
DEFINE_SIMPLE_UNOP(imag);

// binary in terms of total arguments but unary in lattice arguments
static accelerator_inline ComplexD cgpt_pow(const ComplexD & z, double y) { return pow(z,y); };
static accelerator_inline ComplexF cgpt_pow(const ComplexF & z, double y) { return pow(z,(float)y); };
static accelerator_inline ComplexD cgpt_mod(const ComplexD & z, double y) { return ComplexD(fmod(z.real(),y),fmod(z.imag(),y)); };
static accelerator_inline ComplexF cgpt_mod(const ComplexF & z, double y) { return ComplexF(fmod(z.real(),(float)y),fmod(z.imag(),(float)y)); };
static accelerator_inline ComplexD cgpt_relu(const ComplexD & z, double y) { return (z.real() > 0) ? z : y*z; };
static accelerator_inline ComplexF cgpt_relu(const ComplexF & z, double y) { return (z.real() > 0) ? z : ((float)y)*z; };
static accelerator_inline ComplexD cgpt_drelu(const ComplexD & z, double y) { return (z.real() > 0) ? 1 : y; };
static accelerator_inline ComplexF cgpt_drelu(const ComplexF & z, double y) { return (z.real() > 0) ? 1 : ((float)y); };

#define DEFINE_BINOP_REAL(name)						\
  template <class scalar>						\
  struct name ## _functor {						\
    double y;								\
    accelerator name ## _functor(double _y) : y(_y){};			\
    accelerator scalar operator()(const scalar &a) const { return name(a, y); } \
  };									\
  template <class S, class V>						\
  accelerator_inline Grid_simd<S, V> name(const Grid_simd<S, V> &r, double y) { \
    return SimdApply(name ## _functor<S>(y), r);			\
  }									\
  template<class obj> Lattice<obj> name(const Lattice<obj> &rhs_i,RealD y){ \
    Lattice<obj> ret_i(rhs_i.Grid());					\
    autoView( rhs, rhs_i, AcceleratorRead);				\
    autoView( ret, ret_i, AcceleratorWrite);				\
    ret.Checkerboard() = rhs.Checkerboard();				\
    static constexpr int n_elements = GridTypeMapper<obj>::count;	\
    accelerator_for(ss,rhs_i.Grid()->oSites(),obj::Nsimd(),{		\
	for (int e=0;e<n_elements;e++) {				\
	  auto x = coalescedReadElement(rhs[ss], e);			\
	  coalescedWriteElement(ret[ss], name(x,y), e);			\
	}								\
      });								\
    return ret_i;							\
  }

DEFINE_BINOP_REAL(cgpt_drelu);
DEFINE_BINOP_REAL(cgpt_relu);
DEFINE_BINOP_REAL(cgpt_mod);
DEFINE_BINOP_REAL(cgpt_pow);
