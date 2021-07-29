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
template<typename T1, typename T2, int unary_expr, typename Enabled = void>
struct MultiplicationTable {};

// Helper
template <class T> struct isNotSinglet : std::true_type {};
template <class T> struct isNotSinglet<iSinglet<T>> : std::false_type {};

template<typename T,typename vtype>
struct tensorMultType {
  typedef decltype(T()*vtype()) type;
};

template<typename inner, typename vtype>
struct tensorMultType<iScalar<inner>,vtype> {
  typedef iScalar<typename tensorMultType<inner,vtype>::type> type;
};

template<typename inner, int n, typename vtype>
struct tensorMultType<iVector<inner,n>,vtype> {
  typedef iVector<typename tensorMultType<inner,vtype>::type,n> type;
};

template<typename inner, int n, typename vtype>
struct tensorMultType<iMatrix<inner,n>,vtype> {
  typedef iMatrix<typename tensorMultType<inner,vtype>::type,n> type;
};

// General
template<typename T1, typename vtype2, int unary_expr>
struct MultiplicationTable<T1,iSinglet<vtype2>,unary_expr> {
  typedef typename tensorMultType<T1,vtype2>::type result_type;
  typedef iSinglet<vtype2> T2;
  static constexpr int n_elements = GridTypeMapper<result_type>::count;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    ac.coalescedWriteElement(osite, coalescedReadElement(a, e) * coalescedReadElement(b, 0), e);
  }
};

template<typename vtype1, typename T2, int unary_expr>
struct MultiplicationTable<iSinglet<vtype1>,T2,unary_expr,
			   typename std::enable_if<isNotSinglet<T2>::value>::type> {
  typedef typename tensorMultType<T2,vtype1>::type result_type;
  typedef iSinglet<vtype1> T1;
  static constexpr int n_elements = GridTypeMapper<result_type>::count;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    ac.coalescedWriteElement(osite, coalescedReadElement(b, e) * coalescedReadElement(a, 0), e);
  }
};

// MSinglet x MSinglet
template<int n, typename vtype1, typename vtype2>
struct MultiplicationTable<iMatrix<iSinglet<vtype1>,n>, iMatrix<iSinglet<vtype2>,n>, BIT_COLORTRACE> {
  typedef iMatrix<iSinglet<vtype1>,n> T1;
  typedef iMatrix<iSinglet<vtype2>,n> T2;
  typedef iSinglet<decltype(vtype1()*vtype2())> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    DEF_z();
    for (int i=0;i<n;i++)
      for (int j=0;j<n;j++)
	v += coalescedReadElement(a,i,j) * coalescedReadElement(b,j,i);
    ac.coalescedWriteSinglet(osite, v);
  }
};

template<int n, typename vtype1, typename vtype2>
struct MultiplicationTable<iMatrix<iSinglet<vtype1>,n>, iMatrix<iSinglet<vtype2>,n>, BIT_COLORTRACE|BIT_SPINTRACE> :
  MultiplicationTable<iMatrix<iSinglet<vtype1>,n>, iMatrix<iSinglet<vtype2>,n>, BIT_COLORTRACE> {};

template<int n, typename vtype1, typename vtype2, int unary_expr>
struct MultiplicationTable<iMatrix<iSinglet<vtype1>,n>, iMatrix<iSinglet<vtype2>,n>, unary_expr> {
  typedef iMatrix<iSinglet<vtype1>,n> T1;
  typedef iMatrix<iSinglet<vtype2>,n> T2;
  typedef iMatrix<iSinglet<decltype(vtype1()*vtype2())>,n> result_type;
  static constexpr int n_elements = n*n;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int i = e / n;
    int j = e % n;
    DEF_z();
    for (int l=0;l<n;l++)
      v += coalescedReadElement(a,i,l) * coalescedReadElement(b,l,j);
    ac.coalescedWrite(osite, i, j, v);
  }
};

/// VSinglet x VSinglet
template<int n, typename vtype1, typename vtype2, int unary_expr>
struct MultiplicationTable<iVector<iSinglet<vtype1>,n>, iVector<iSinglet<vtype2>,n>, unary_expr> {
  typedef iVector<iSinglet<vtype1>,n> T1;
  typedef iVector<iSinglet<vtype2>,n> T2;
  typedef iVector<iSinglet<decltype(vtype1()*vtype2())>,n> result_type;
  static constexpr int n_elements = n;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, i, coalescedReadElement(a,i) * coalescedReadElement(b,i));
  }
};

// MSinglet x VSinglet
template<int n, typename vtype1, typename vtype2, int unary_expr>
struct MultiplicationTable<iMatrix<iSinglet<vtype1>,n>, iVector<iSinglet<vtype2>,n>, unary_expr> {
  typedef iMatrix<iSinglet<vtype1>,n> T1;
  typedef iVector<iSinglet<vtype2>,n> T2;
  typedef iVector<iSinglet<decltype(vtype1()*vtype2())>,n> result_type;
  static constexpr int n_elements = n;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    DEF_z();
    for (int j=0;j<n;j++)
      v += coalescedReadElement(a,i,j) * coalescedReadElement(b,j);
    ac.coalescedWrite(osite, i, v);
  }
};

// MColor x MColor
template<int n, typename vtype1, typename vtype2, int unary_expr>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype1,n>>>, iScalar<iScalar<iMatrix<vtype2,n>>>, unary_expr> {
  typedef iScalar<iScalar<iMatrix<vtype1,n>>> T1;
  typedef iScalar<iScalar<iMatrix<vtype2,n>>> T2;
  typedef iScalar<iScalar<iMatrix<decltype(vtype1()*vtype2()),n>>> result_type;

#ifdef GRID_HAS_ACCELERATOR
  static constexpr int n_elements = n*n;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int i = e / n;
    int j = e % n;
    DEF_z();
    for (int l=0;l<n;l++)
      v += coalescedReadElement(a,i,l) * coalescedReadElement(b,l,j);
    ac.coalescedWrite(osite, i, j, v);
  }
#else
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    ac.coalescedWrite(osite, coalescedRead(a)*coalescedRead(b));
  }
#endif
};

template<int n, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype1,n>>>, iScalar<iScalar<iMatrix<vtype2,n>>>, BIT_COLORTRACE> {
  typedef iScalar<iScalar<iMatrix<vtype1,n>>> T1;
  typedef iScalar<iScalar<iMatrix<vtype2,n>>> T2;
  typedef iSinglet<decltype(vtype1()*vtype2())> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    DEF_z();
    for (int i=0;i<n;i++)
      for (int j=0;j<n;j++)
	v += coalescedReadElement(a,i,j) * coalescedReadElement(b,j,i);
    ac.coalescedWriteSinglet(osite, v);
  }
};

template<int n, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype1,n>>>, iScalar<iScalar<iMatrix<vtype2,n>>>, BIT_COLORTRACE|BIT_SPINTRACE> :
  MultiplicationTable<iScalar<iScalar<iMatrix<vtype1,n>>>, iScalar<iScalar<iMatrix<vtype2,n>>>, BIT_COLORTRACE> {};

// MColor x VColor
template<int n, typename vtype1, typename vtype2, int unary_expr>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype1,n>>>, iScalar<iScalar<iVector<vtype2,n>>>, unary_expr> {
  typedef iScalar<iScalar<iMatrix<vtype1,n>>> T1;
  typedef iScalar<iScalar<iVector<vtype2,n>>> T2;
  typedef iScalar<iScalar<iVector<decltype(vtype1()*vtype2()),n>>> result_type;
  static constexpr int n_elements = n;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    DEF_z();
    for (int j=0;j<n;j++)
      v += coalescedReadElement(a,i,j) * coalescedReadElement(b,j);
    ac.coalescedWrite(osite, i, v);
  }
};

// MSpin x MSpin
template<int n, typename vtype1, typename vtype2, int unary_expr>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype1>,n>>, iScalar<iMatrix<iScalar<vtype2>,n>>, unary_expr> {
  typedef iScalar<iMatrix<iScalar<vtype1>,n>> T1;
  typedef iScalar<iMatrix<iScalar<vtype2>,n>> T2;
  typedef iScalar<iMatrix<iScalar<decltype(vtype1()*vtype2())>,n>> result_type;
#ifdef GRID_HAS_ACCELERATOR
  static constexpr int n_elements = n*n;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int i = e / n;
    int j = e % n;
    DEF_z();
    for (int l=0;l<n;l++)
      v += coalescedReadElement(a,i,l) * coalescedReadElement(b,l,j);
    ac.coalescedWrite(osite, i, j, v);
  }
#else
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    ac.coalescedWrite(osite, coalescedRead(a)*coalescedRead(b));
  }
#endif
};

template<int n, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype1>,n>>, iScalar<iMatrix<iScalar<vtype2>,n>>, BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iScalar<vtype1>,n>> T1;
  typedef iScalar<iMatrix<iScalar<vtype2>,n>> T2;
  typedef iSinglet<decltype(vtype1()*vtype2())> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    DEF_z();
    for (int i=0;i<n;i++)
      for (int j=0;j<n;j++)
	v += coalescedReadElement(a,i,j) * coalescedReadElement(b,j,i);
    ac.coalescedWriteSinglet(osite, v);
  }
};

template<int n, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype1>,n>>, iScalar<iMatrix<iScalar<vtype2>,n>>, BIT_SPINTRACE|BIT_COLORTRACE> :
  MultiplicationTable<iScalar<iMatrix<iScalar<vtype1>,n>>, iScalar<iMatrix<iScalar<vtype2>,n>>, BIT_SPINTRACE> {};


// MSpin x VSpin
template<int n, typename vtype1, typename vtype2, int unary_expr>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype1>,n>>, iScalar<iVector<iScalar<vtype2>,n>>, unary_expr> {
  typedef iScalar<iMatrix<iScalar<vtype1>,n>> T1;
  typedef iScalar<iVector<iScalar<vtype2>,n>> T2;
  typedef iScalar<iVector<iScalar<decltype(vtype1()*vtype2())>,n>> result_type;
  static constexpr int n_elements = n;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    DEF_z();
    for (int j=0;j<n;j++)
      v += coalescedReadElement(a,i,j) * coalescedReadElement(b,j);
    ac.coalescedWrite(osite, i, v);
  }
};

// Gamma x VSpin
template<int n, typename vtype2, int unary_expr>
struct MultiplicationTable<Gamma, iScalar<iVector<iScalar<vtype2>,n>>, unary_expr> {
  typedef Gamma T1;
  typedef iScalar<iVector<iScalar<vtype2>,n>> T2;
  typedef iScalar<iVector<iScalar<vtype2>,n>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, a * coalescedRead(b));
  }
};

// Gamma x MSpin
template<int n, typename vtype2, int unary_expr>
struct MultiplicationTable<Gamma, iScalar<iMatrix<iScalar<vtype2>,n>>, unary_expr> {
  typedef Gamma T1;
  typedef iScalar<iMatrix<iScalar<vtype2>,n>> T2;
  typedef iScalar<iMatrix<iScalar<vtype2>,n>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, a * coalescedRead(b));
  }
};

template<int n, typename vtype2>
struct MultiplicationTable<Gamma, iScalar<iMatrix<iScalar<vtype2>,n>>, BIT_SPINTRACE> {
  typedef Gamma T1;
  typedef iScalar<iMatrix<iScalar<vtype2>,n>> T2;
  typedef iScalar<iScalar<iScalar<vtype2>>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, traceIndex<SpinIndex>(a * coalescedRead(b)));
  }
};

template<int n, typename vtype2>
struct MultiplicationTable<Gamma, iScalar<iMatrix<iScalar<vtype2>,n>>, BIT_SPINTRACE|BIT_COLORTRACE> :
  MultiplicationTable<Gamma, iScalar<iMatrix<iScalar<vtype2>,n>>, BIT_SPINTRACE> {};


// MSpin x Gamma
template<int n, typename vtype1, int unary_expr>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype1>,n>>, Gamma, unary_expr> {
  typedef iScalar<iMatrix<iScalar<vtype1>,n>> T1;
  typedef Gamma T2;
  typedef iScalar<iMatrix<iScalar<vtype1>,n>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, coalescedRead(a) * b);
  }
};

template<int n, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype2>,n>>, Gamma, BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iScalar<vtype2>,n>> T1;
  typedef Gamma T2;
  typedef iScalar<iScalar<iScalar<vtype2>>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, traceIndex<SpinIndex>(coalescedRead(a) * b));
  }
};

template<int n, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype2>,n>>, Gamma, BIT_SPINTRACE|BIT_COLORTRACE> :
  MultiplicationTable<iScalar<iMatrix<iScalar<vtype2>,n>>, Gamma, BIT_SPINTRACE> {};


// Gamma x VSpinColor
template<int ns, int nc, typename vtype2, int unary_expr>
struct MultiplicationTable<Gamma, iScalar<iVector<iVector<vtype2,nc>,ns>>, unary_expr> {
  typedef Gamma T1;
  typedef iScalar<iVector<iVector<vtype2,nc>,ns>> T2;
  typedef iScalar<iVector<iVector<vtype2,nc>,ns>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, a * coalescedRead(b));
  }
};

// Gamma x MSpinColor
template<int ns, int nc, typename vtype2>
struct MultiplicationTable<Gamma, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, 0> {
  typedef Gamma T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, a * coalescedRead(b));
  }
};

template<int ns, int nc, typename vtype2>
struct MultiplicationTable<Gamma, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, BIT_SPINTRACE> {
  typedef Gamma T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iScalar<iScalar<iMatrix<vtype2,nc>>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, traceIndex<SpinIndex>(a * coalescedRead(b)));
  }
};

template<int ns, int nc, typename vtype2>
struct MultiplicationTable<Gamma, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, BIT_COLORTRACE> {
  typedef Gamma T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iScalar<iMatrix<iScalar<vtype2>,ns>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, traceIndex<ColourIndex>(a * coalescedRead(b)));
  }
};

template<int ns, int nc, typename vtype2>
struct MultiplicationTable<Gamma, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, BIT_SPINTRACE|BIT_COLORTRACE> {
  typedef Gamma T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iScalar<iScalar<iScalar<vtype2>>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, trace(a * coalescedRead(b)));
  }
};

// MSpinColor x Gamma
template<int ns, int nc, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>,Gamma, 0> {
  typedef Gamma T2;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, coalescedRead(a) * b);
  }
};

template<int ns, int nc, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>,Gamma, BIT_SPINTRACE> {
  typedef Gamma T2;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T1;
  typedef iScalar<iScalar<iMatrix<vtype2,nc>>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, traceIndex<SpinIndex>(coalescedRead(a) * b));
  }
};

template<int ns, int nc, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>,Gamma, BIT_COLORTRACE> {
  typedef Gamma T2;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T1;
  typedef iScalar<iMatrix<iScalar<vtype2>,ns>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, traceIndex<ColourIndex>(coalescedRead(a) * b));
  }
};

template<int ns, int nc, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>,Gamma, BIT_SPINTRACE|BIT_COLORTRACE> {
  typedef Gamma T2;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T1;
  typedef iScalar<iScalar<iScalar<vtype2>>> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int i) {
    ac.coalescedWrite(osite, trace(coalescedRead(a) * b));
  }
};

// MColor x VSpinColor
template<int ns, int nc, typename vtype1, typename vtype2, int unary_expr>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype1,nc>>>, iScalar<iVector<iVector<vtype2,nc>,ns>>, unary_expr> {
  typedef iScalar<iScalar<iMatrix<vtype1,nc>>> T1;
  typedef iScalar<iVector<iVector<vtype2,nc>,ns>> T2;
  typedef iScalar<iVector<iVector<decltype(vtype1()*vtype2()),nc>,ns>> result_type;
  static constexpr int n_elements = nc*ns;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int is = e / nc;
    int ic = e % nc;
    DEF_z();
    for (int jc=0;jc<nc;jc++)
      v += coalescedReadElement(a,ic,jc) * coalescedReadElement(b,is,jc);
    ac.coalescedWrite(osite, is, ic, v);
  }
};

// MColor x MSpinColor
template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype1,nc>>>, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, 0> {
  typedef iScalar<iScalar<iMatrix<vtype1,nc>>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iScalar<iMatrix<iMatrix<decltype(vtype1()*vtype2()),nc>,ns>> result_type;
  static constexpr int n_elements = nc*ns*nc*ns;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int ie = e % (nc * ns);
    int je = e / (nc * ns);
      
    int is = ie / nc;
    int ic = ie % nc;

    int js = je / nc;
    int jc = je % nc;
    
    DEF_z();
    for (int lc=0;lc<nc;lc++)
      v += coalescedReadElement(a,ic,lc) * coalescedReadElement(b,is,js,lc,jc);
    ac.coalescedWrite(osite, is, js, ic, jc, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype1,nc>>>, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, BIT_COLORTRACE> {
  typedef iScalar<iScalar<iMatrix<vtype1,nc>>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iScalar<iMatrix<iScalar<decltype(vtype1()*vtype2())>,ns>> result_type;
  static constexpr int n_elements = ns*ns;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int is = e / ns;
    int js = e % ns;
    
    DEF_z();
    for (int ic=0;ic<nc;ic++)
      for (int lc=0;lc<nc;lc++)
	v += coalescedReadElement(a,ic,lc) * coalescedReadElement(b,is,js,lc,ic);
    ac.coalescedWrite(osite, is, js, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype1,nc>>>, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, BIT_SPINTRACE> {
  typedef iScalar<iScalar<iMatrix<vtype1,nc>>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iScalar<iScalar<iMatrix<decltype(vtype1()*vtype2()),nc>>> result_type;
  static constexpr int n_elements = nc*nc;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int ic = e % nc;
    int jc = e / nc;

    DEF_z();
    for (int ls=0;ls<ns;ls++)
      for (int lc=0;lc<nc;lc++)
	v += coalescedReadElement(a,ic,lc) * coalescedReadElement(b,ls,ls,lc,jc);
    ac.coalescedWrite(osite, ic, jc, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype1,nc>>>, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, BIT_SPINTRACE|BIT_COLORTRACE> {
  typedef iScalar<iScalar<iMatrix<vtype1,nc>>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iSinglet<decltype(vtype1()*vtype2())> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    DEF_z();
    for (int ic=0;ic<nc;ic++)
      for (int ls=0;ls<ns;ls++)
	for (int lc=0;lc<nc;lc++)
	  v += coalescedReadElement(a,ic,lc) * coalescedReadElement(b,ls,ls,lc,ic);
    ac.coalescedWriteSinglet(osite, v);
  }
};

// MSpinColor x MColor
template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype1,nc>,ns>>, iScalar<iScalar<iMatrix<vtype2,nc>>>, 0> {
  typedef iScalar<iMatrix<iMatrix<vtype1,nc>,ns>> T1;
  typedef iScalar<iScalar<iMatrix<vtype2,nc>>> T2;
  typedef iScalar<iMatrix<iMatrix<decltype(vtype1()*vtype2()),nc>,ns>> result_type;
  static constexpr int n_elements = nc*ns*nc*ns;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int ie = e % (nc * ns);
    int je = e / (nc * ns);
      
    int is = ie / nc;
    int ic = ie % nc;

    int js = je / nc;
    int jc = je % nc;
    
    DEF_z();
    for (int lc=0;lc<nc;lc++)
      v += coalescedReadElement(a,is,js,ic,lc) * coalescedReadElement(b,lc,jc);
    ac.coalescedWrite(osite, is, js, ic, jc, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype1,nc>,ns>>, iScalar<iScalar<iMatrix<vtype2,nc>>>, BIT_COLORTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype1,nc>,ns>> T1;
  typedef iScalar<iScalar<iMatrix<vtype2,nc>>> T2;
  typedef iScalar<iMatrix<iScalar<decltype(vtype1()*vtype2())>,ns>> result_type;
  static constexpr int n_elements = ns*ns;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int is = e / ns;
    int js = e % ns;
    
    DEF_z();
    for (int lc=0;lc<nc;lc++)
      for (int ic=0;ic<nc;ic++)
	v += coalescedReadElement(a,is,js,ic,lc) * coalescedReadElement(b,lc,ic);
    ac.coalescedWrite(osite, is, js, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype1,nc>,ns>>, iScalar<iScalar<iMatrix<vtype2,nc>>>, BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype1,nc>,ns>> T1;
  typedef iScalar<iScalar<iMatrix<vtype2,nc>>> T2;
  typedef iScalar<iScalar<iMatrix<decltype(vtype1()*vtype2()),nc>>> result_type;
  static constexpr int n_elements = nc*nc;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int ic = e / nc;
    int jc = e % nc;
    
    DEF_z();
    for (int lc=0;lc<nc;lc++)
      for (int is=0;is<ns;is++)
	v += coalescedReadElement(a,is,is,ic,lc) * coalescedReadElement(b,lc,jc);
    ac.coalescedWrite(osite, ic, jc, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype1,nc>,ns>>, iScalar<iScalar<iMatrix<vtype2,nc>>>, BIT_SPINTRACE|BIT_COLORTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype1,nc>,ns>> T1;
  typedef iScalar<iScalar<iMatrix<vtype2,nc>>> T2;
  typedef iSinglet<decltype(vtype1()*vtype2())> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    
    DEF_z();
    for (int lc=0;lc<nc;lc++)
      for (int is=0;is<ns;is++)
	for (int ic=0;ic<nc;ic++)
	  v += coalescedReadElement(a,is,is,ic,lc) * coalescedReadElement(b,lc,ic);
    ac.coalescedWriteSinglet(osite, v);
  }
};


// MSpin x VSpinColor
template<int ns, int nc, typename vtype1, typename vtype2, int unary_expr>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype1>,ns>>, iScalar<iVector<iVector<vtype2,nc>,ns>>, unary_expr> {
  typedef iScalar<iMatrix<iScalar<vtype1>,ns>> T1;
  typedef iScalar<iVector<iVector<vtype2,nc>,ns>> T2;
  typedef iScalar<iVector<iVector<decltype(vtype1()*vtype2()),nc>,ns>> result_type;
  static constexpr int n_elements = nc*ns;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int is = e / nc;
    int ic = e % nc;
    DEF_z();
    for (int js=0;js<ns;js++)
      v += coalescedReadElement(a,is,js) * coalescedReadElement(b,js,ic);
    ac.coalescedWrite(osite, is, ic, v);
  }
};

// MSpin x MSpinColor
template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype1>,ns>>, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, 0> {
  typedef iScalar<iMatrix<iScalar<vtype1>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iScalar<iMatrix<iMatrix<decltype(vtype1()*vtype2()),nc>,ns>> result_type;
  static constexpr int n_elements = nc*ns*nc*ns;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int ie = e % (nc * ns);
    int je = e / (nc * ns);
      
    int is = ie / nc;
    int ic = ie % nc;

    int js = je / nc;
    int jc = je % nc;
    
    DEF_z();
    for (int ls=0;ls<ns;ls++)
      v += coalescedReadElement(a,is,ls) * coalescedReadElement(b,ls,js,ic,jc);
    ac.coalescedWrite(osite, is, js, ic, jc, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype1>,ns>>, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, BIT_COLORTRACE> {
  typedef iScalar<iMatrix<iScalar<vtype1>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iScalar<iMatrix<iScalar<decltype(vtype1()*vtype2())>,ns>> result_type;
  static constexpr int n_elements = ns*ns;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int is = e / ns;
    int js = e % ns;
    
    DEF_z();
    for (int ic=0;ic<nc;ic++)
      for (int ls=0;ls<ns;ls++)
	v += coalescedReadElement(a,is,ls) * coalescedReadElement(b,ls,js,ic,ic);
    ac.coalescedWrite(osite, is, js, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype1>,ns>>, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iScalar<vtype1>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iScalar<iScalar<iMatrix<decltype(vtype1()*vtype2()),nc>>> result_type;
  static constexpr int n_elements = nc*nc;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int ic = e / nc;
    int jc = e % nc;
    
    DEF_z();
    for (int ls=0;ls<ns;ls++)
      for (int is=0;is<ns;is++)
	v += coalescedReadElement(a,is,ls) * coalescedReadElement(b,ls,is,ic,jc);
    ac.coalescedWrite(osite, ic, jc, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype1>,ns>>, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, BIT_SPINTRACE|BIT_COLORTRACE> {
  typedef iScalar<iMatrix<iScalar<vtype1>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iSinglet<decltype(vtype1()*vtype2())> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    
    DEF_z();
    for (int ls=0;ls<ns;ls++)
      for (int is=0;is<ns;is++)
	for (int ic=0;ic<nc;ic++)
	  v += coalescedReadElement(a,is,ls) * coalescedReadElement(b,ls,is,ic,ic);
    ac.coalescedWriteSinglet(osite, v);
  }
};



// MSpinColor x MSpin
template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype1,nc>,ns>>, iScalar<iMatrix<iScalar<vtype2>,ns>>, 0> {
  typedef iScalar<iMatrix<iMatrix<vtype1,nc>,ns>> T1;
  typedef iScalar<iMatrix<iScalar<vtype2>,ns>> T2;
  typedef iScalar<iMatrix<iMatrix<decltype(vtype1()*vtype2()),nc>,ns>> result_type;
  static constexpr int n_elements = nc*ns*nc*ns;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int ie = e % (nc * ns);
    int je = e / (nc * ns);
      
    int is = ie / nc;
    int ic = ie % nc;

    int js = je / nc;
    int jc = je % nc;
    
    DEF_z();
    for (int ls=0;ls<ns;ls++)
      v += coalescedReadElement(a,is,ls,ic,jc) * coalescedReadElement(b,ls,js);
    ac.coalescedWrite(osite, is, js, ic, jc, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype1,nc>,ns>>, iScalar<iMatrix<iScalar<vtype2>,ns>>, BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype1,nc>,ns>> T1;
  typedef iScalar<iMatrix<iScalar<vtype2>,ns>> T2;
  typedef iScalar<iScalar<iMatrix<decltype(vtype1()*vtype2()),nc>>> result_type;
  static constexpr int n_elements = nc*nc;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int ic = e / nc;
    int jc = e % nc;
    
    DEF_z();
    for (int ls=0;ls<ns;ls++)
      for (int is=0;is<ns;is++)
	v += coalescedReadElement(a,is,ls,ic,jc) * coalescedReadElement(b,ls,is);
    ac.coalescedWrite(osite, ic, jc, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype1,nc>,ns>>, iScalar<iMatrix<iScalar<vtype2>,ns>>, BIT_COLORTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype1,nc>,ns>> T1;
  typedef iScalar<iMatrix<iScalar<vtype2>,ns>> T2;
  typedef iScalar<iMatrix<iScalar<decltype(vtype1()*vtype2())>,ns>> result_type;
  static constexpr int n_elements = ns*ns;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int is = e / ns;
    int js = e % ns;

    DEF_z();
    for (int ls=0;ls<ns;ls++)
      for (int ic=0;ic<nc;ic++)
	v += coalescedReadElement(a,is,ls,ic,ic) * coalescedReadElement(b,ls,js);
    ac.coalescedWrite(osite, is, js, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype1,nc>,ns>>, iScalar<iMatrix<iScalar<vtype2>,ns>>, BIT_COLORTRACE|BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype1,nc>,ns>> T1;
  typedef iScalar<iMatrix<iScalar<vtype2>,ns>> T2;
  typedef iSinglet<decltype(vtype1()*vtype2())> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    DEF_z();
    for (int ls=0;ls<ns;ls++)
      for (int ic=0;ic<nc;ic++)
	for (int is=0;is<ns;is++)
	  v += coalescedReadElement(a,is,ls,ic,ic) * coalescedReadElement(b,ls,is);
    ac.coalescedWriteSinglet(osite, v);
  }
};

// MSpinColor x MSpinColor
template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype1,nc>,ns>>, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, 0> {
  typedef iScalar<iMatrix<iMatrix<vtype1,nc>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iScalar<iMatrix<iMatrix<decltype(vtype1()*vtype2()),nc>,ns>> result_type;
  static constexpr int n_elements = ns*ns;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int is = e / ns; // % QP4 benchmark: this order better, L2 prefetching current 212, 217 GB/s
    int js = e % ns;

    typedef iMatrix<decltype(vtype1()*vtype2()),nc> O_t;
    DEF_o(O_t);
    for (int ls=0;ls<ns;ls++) {
      for (int ic=0;ic<nc;ic++) {
	for (int lc=0;lc<nc;lc++) {
	  auto x = coalescedReadElement(a,is,ls,ic,lc);
	  for (int jc=0;jc<nc;jc++) {
	    v(ic,jc) += x * coalescedReadElement(b,ls,js,lc,jc);
	  }
	}
      }
    }

    for (int ic=0;ic<nc;ic++) {
      for (int jc=0;jc<nc;jc++) {
	ac.coalescedWrite(osite, is, js, ic, jc, v(ic,jc));
      }
    }
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype1,nc>,ns>>, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype1,nc>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iScalar<iScalar<iMatrix<decltype(vtype1()*vtype2()),nc>>> result_type;
  static constexpr int n_elements = nc*nc;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int ic = e / nc;
    int jc = e % nc;
    
    DEF_z();
    for (int ls=0;ls<ns;ls++)
      for (int lc=0;lc<nc;lc++)
	for (int is=0;is<ns;is++)
	  v += coalescedReadElement(a,is,ls,ic,lc) * coalescedReadElement(b,ls,is,lc,jc);
    ac.coalescedWrite(osite, ic, jc, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype1,nc>,ns>>, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, BIT_COLORTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype1,nc>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iScalar<iMatrix<iScalar<decltype(vtype1()*vtype2())>,ns>> result_type;
  static constexpr int n_elements = ns*ns;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int is = e / ns;
    int js = e % ns;
    
    DEF_z();
    for (int ls=0;ls<ns;ls++)
      for (int lc=0;lc<nc;lc++)
	for (int ic=0;ic<nc;ic++)
	  v += coalescedReadElement(a,is,ls,ic,lc) * coalescedReadElement(b,ls,js,lc,ic);
    ac.coalescedWrite(osite, is, js, v);
  }
};

template<int ns, int nc, typename vtype1, typename vtype2>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype1,nc>,ns>>, iScalar<iMatrix<iMatrix<vtype2,nc>,ns>>, BIT_COLORTRACE|BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype1,nc>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype2,nc>,ns>> T2;
  typedef iSinglet<decltype(vtype1()*vtype2())> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    
    DEF_z();
    for (int ls=0;ls<ns;ls++)
      for (int lc=0;lc<nc;lc++)
	for (int ic=0;ic<nc;ic++)
	  for (int is=0;is<ns;is++)
	    v += coalescedReadElement(a,is,ls,ic,lc) * coalescedReadElement(b,ls,is,lc,ic);
    ac.coalescedWriteSinglet(osite, v);
  }
};

// MSpinColor x VSpinColor
template<int ns, int nc, typename vtype1, typename vtype2, int unary_expr>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype1,nc>,ns>>, iScalar<iVector<iVector<vtype2,nc>,ns>>, unary_expr> {
  typedef iScalar<iMatrix<iMatrix<vtype1,nc>,ns>> T1;
  typedef iScalar<iVector<iVector<vtype2,nc>,ns>> T2;
  typedef iScalar<iVector<iVector<decltype(vtype1()*vtype2()),nc>,ns>> result_type;
  static constexpr int n_elements = nc*ns;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    int is = e / nc;
    int ic = e % nc;
    
    DEF_z();
    for (int js=0;js<ns;js++)
      for (int jc=0;jc<nc;jc++)
	v += coalescedReadElement(a,is,js,ic,jc) * coalescedReadElement(b,js,jc);
    ac.coalescedWrite(osite, is, ic, v);
  }
};

// and reverse
template<typename T1, typename T2, int unary_expr, bool rev>
struct MultiplicationTableRev {};

template<typename T1, typename T2, int unary_expr>
struct MultiplicationTableRev<T1,T2,unary_expr,false> {
  typedef MultiplicationTable<T1,T2,unary_expr> type;
  typedef typename type::result_type result_type;
  static constexpr int n_elements = type::n_elements;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    return type::eval(ac,osite,a,b,e);
  }
};

template<typename T1, typename T2, int unary_expr>
struct MultiplicationTableRev<T1,T2,unary_expr,true> {
  typedef MultiplicationTable<T2,T1,unary_expr> type;
  typedef typename type::result_type result_type;
  static constexpr int n_elements = type::n_elements;
  template<typename Accumulator> static accelerator_inline void eval(Accumulator & ac, uint64_t osite, const T1 & a, const T2 & b, int e) {
    return type::eval(ac,osite,b,a,e);
  }
};
