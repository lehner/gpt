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


// unary operators acting on complex numbers both on CPU and GPU
#define UNARY(func)							\
  template<class obj> accelerator_inline auto func(const iScalar<obj> &z) -> iScalar<obj> \
  {									\
    iScalar<obj> ret;							\
    ret._internal = func( (z._internal));				\
    return ret;								\
  }									\
  template<class obj,int N> accelerator_inline auto func(const iVector<obj,N> &z) -> iVector<obj,N>	\
  {									\
    iVector<obj,N> ret;							\
    for(int c1=0;c1<N;c1++){						\
      ret._internal[c1] = func( (z._internal[c1]));			\
    }									\
    return ret;								\
  }									\
  template<class obj,int N> accelerator_inline auto func(const iMatrix<obj,N> &z) -> iMatrix<obj,N>	\
  {									\
    iMatrix<obj,N> ret;							\
    for(int c1=0;c1<N;c1++){						\
      for(int c2=0;c2<N;c2++){						\
	ret._internal[c1][c2] = func( (z._internal[c1][c2]));		\
      }}								\
    return ret;								\
  }


#define BINARY_RSCALAR(func,scal)					\
  template<class obj> accelerator_inline iScalar<obj> func(const iScalar<obj> &z,scal y) \
  {									\
    iScalar<obj> ret;							\
    ret._internal = func(z._internal,y);				\
    return ret;								\
  }									\
  template<class obj,int N> accelerator_inline iVector<obj,N> func(const iVector<obj,N> &z,scal y) \
  {									\
    iVector<obj,N> ret;							\
    for(int c1=0;c1<N;c1++){						\
      ret._internal[c1] = func(z._internal[c1],y);			\
    }									\
    return ret;								\
  }									\
  template<class obj,int N> accelerator_inline  iMatrix<obj,N> func(const iMatrix<obj,N> &z, scal y) \
  {									\
    iMatrix<obj,N> ret;							\
    for(int c1=0;c1<N;c1++){						\
      for(int c2=0;c2<N;c2++){						\
	ret._internal[c1][c2] = func(z._internal[c1][c2],y);		\
      }}								\
    return ret;								\
  }

#define DEFINE_SIMPLE_UNOP(tag)						\
  template <class scalar>						\
  struct cgpt_ ## tag ## _functor {					\
    accelerator scalar operator()(const scalar &a) const { return tag(a); } \
  };									\
  static accelerator_inline ComplexD _cgpt_ ## tag(const ComplexD & z) { return tag(z); };	\
  static accelerator_inline ComplexF _cgpt_ ## tag(const ComplexF & z) { return tag(z); };	\
  template <class S, class V>						\
    accelerator_inline Grid_simd<S, V> _cgpt_ ## tag (const Grid_simd<S, V> &r) { \
      return SimdApply(cgpt_ ## tag ## _functor<S>(), r);		\
  }									\
  UNARY(_cgpt_ ## tag );						\
  GridUnopClass(cgpt_unary_ ## tag , _cgpt_ ## tag(a));			\
  GRID_DEF_UNOP(cgpt_ ## tag , cgpt_unary_ ## tag);

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

// binary in terms of total arguments but unary in lattice arguments
static accelerator_inline ComplexD _cgpt_pow(const ComplexD & z, double y) { return pow(z,y); };
static accelerator_inline ComplexF _cgpt_pow(const ComplexF & z, double y) { return pow(z,(float)y); };
template <class scalar>
struct cgpt_pow_functor {
  double y;
  accelerator cgpt_pow_functor(double _y) : y(_y){};
  accelerator scalar operator()(const scalar &a) const { return _cgpt_pow(a, y); }
};
template <class S, class V>
accelerator_inline Grid_simd<S, V> cgpt_pow(const Grid_simd<S, V> &r, double y) {
  return SimdApply(cgpt_pow_functor<S>(y), r);
}
BINARY_RSCALAR(cgpt_pow,RealD);
template<class obj> Lattice<obj> cgpt_pow(const Lattice<obj> &rhs_i,RealD y){
  Lattice<obj> ret_i(rhs_i.Grid());
  autoView( rhs, rhs_i, AcceleratorRead);
  autoView( ret, ret_i, AcceleratorWrite);
  ret.Checkerboard() = rhs.Checkerboard();
  accelerator_for(ss,rhs.size(),1,{
      ret[ss]=cgpt_pow(rhs[ss],y);
  });
  return ret_i;
}


static accelerator_inline ComplexD _cgpt_mod(const ComplexD & z, double y) { return ComplexD(fmod(z.real(),y),fmod(z.imag(),y)); };
static accelerator_inline ComplexF _cgpt_mod(const ComplexF & z, double y) { return ComplexF(fmod(z.real(),(float)y),fmod(z.imag(),(float)y)); };
template <class scalar>
struct cgpt_mod_functor {
  double y;
  accelerator cgpt_mod_functor(double _y) : y(_y){};
  accelerator scalar operator()(const scalar &a) const { return _cgpt_mod(a, y); }
};
template <class S, class V>
accelerator_inline Grid_simd<S, V> cgpt_mod(const Grid_simd<S, V> &r, double y) {
  return SimdApply(cgpt_mod_functor<S>(y), r);
}
BINARY_RSCALAR(cgpt_mod,RealD);
template<class obj> Lattice<obj> cgpt_mod(const Lattice<obj> &rhs_i,RealD y){
  Lattice<obj> ret_i(rhs_i.Grid());
  autoView( rhs, rhs_i, AcceleratorRead);
  autoView( ret, ret_i, AcceleratorWrite);
  ret.Checkerboard() = rhs.Checkerboard();
  accelerator_for(ss,rhs.size(),1,{
      ret[ss]=cgpt_mod(rhs[ss],y);
  });
  return ret_i;
}


static accelerator_inline ComplexD _cgpt_relu(const ComplexD & z, double y) { return (z.real() > 0) ? z : y*z; };
static accelerator_inline ComplexF _cgpt_relu(const ComplexF & z, double y) { return (z.real() > 0) ? z : ((float)y)*z; };
template <class scalar>
struct cgpt_relu_functor {
  double y;
  accelerator cgpt_relu_functor(double _y) : y(_y){};
  accelerator scalar operator()(const scalar &a) const { return _cgpt_relu(a, y); }
};
template <class S, class V>
accelerator_inline Grid_simd<S, V> cgpt_relu(const Grid_simd<S, V> &r, double y) {
  return SimdApply(cgpt_relu_functor<S>(y), r);
}
BINARY_RSCALAR(cgpt_relu,RealD);
template<class obj> Lattice<obj> cgpt_relu(const Lattice<obj> &rhs_i,RealD y){
  Lattice<obj> ret_i(rhs_i.Grid());
  autoView( rhs, rhs_i, AcceleratorRead);
  autoView( ret, ret_i, AcceleratorWrite);
  ret.Checkerboard() = rhs.Checkerboard();
  accelerator_for(ss,rhs.size(),1,{
      ret[ss]=cgpt_relu(rhs[ss],y);
  });
  return ret_i;
}


static accelerator_inline ComplexD _cgpt_drelu(const ComplexD & z, double y) { return (z.real() > 0) ? 1 : y; };
static accelerator_inline ComplexF _cgpt_drelu(const ComplexF & z, double y) { return (z.real() > 0) ? 1 : ((float)y); };
template <class scalar>
struct cgpt_drelu_functor {
  double y;
  accelerator cgpt_drelu_functor(double _y) : y(_y){};
  accelerator scalar operator()(const scalar &a) const { return _cgpt_drelu(a, y); }
};
template <class S, class V>
accelerator_inline Grid_simd<S, V> cgpt_drelu(const Grid_simd<S, V> &r, double y) {
  return SimdApply(cgpt_drelu_functor<S>(y), r);
}
BINARY_RSCALAR(cgpt_drelu,RealD);
template<class obj> Lattice<obj> cgpt_drelu(const Lattice<obj> &rhs_i,RealD y){
  Lattice<obj> ret_i(rhs_i.Grid());
  autoView( rhs, rhs_i, AcceleratorRead);
  autoView( ret, ret_i, AcceleratorWrite);
  ret.Checkerboard() = rhs.Checkerboard();
  accelerator_for(ss,rhs.size(),1,{
      ret[ss]=cgpt_drelu(rhs[ss],y);
  });
  return ret_i;
}
