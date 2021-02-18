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
#define typeis(a,at) ( a->type() == typeid(at<vtype>).name() )
#define castas(a,at) ( ((cgpt_Lattice<at<vtype> >*)a)->l )
#define typeOpen(a,at) if (typeis(a,at)) { auto& l ## a = castas(a,at);
#define typeClose() }

template<typename T>
const Lattice<T> cgpt_unary(int unary, const Lattice<T> & l) {
  switch (unary) {
  case 0:
    return l;
  case BIT_TRANS|BIT_CONJ:
    return adj(l);
  case BIT_TRANS:
    return transpose(l);
  case BIT_CONJ:
    return conjugate(l);
  default:
    ERR("Unknown unary %d",unary);
  }
}


template<typename T>
accelerator_inline
auto coalescedReadElement(const T & c, int e) -> decltype(coalescedRead(typename T::vector_type())) {
  typedef typename T::vector_type vCoeff_t;
  vCoeff_t * p = (vCoeff_t*)&c;
  return coalescedRead(p[e]);
}

struct AccumulatorYes {
  
  template<typename T, typename V>
  static accelerator_inline
  void coalescedWriteElement(T & c, const V & v, int e) {
    typedef typename T::vector_type vCoeff_t;
    vCoeff_t * p = (vCoeff_t*)&c;
    V r = v + coalescedReadElement(c, e);
    ::coalescedWrite(p[e], r);
  }

  template<typename T, typename V>
  static accelerator_inline
  void coalescedWrite(T & c, const V & v) {
    V r = v + coalescedRead(c);
    ::coalescedWrite(c, r);
  }

  static constexpr ViewMode AcceleratorWriteMode = AcceleratorWrite;

};

struct AccumulatorNo {
  
  template<typename T, typename V>
  static accelerator_inline
  void coalescedWriteElement(T & c, const V & v, int e) {
    typedef typename T::vector_type vCoeff_t;
    vCoeff_t * p = (vCoeff_t*)&c;
    ::coalescedWrite(p[e], v);
  }

  template<typename T, typename V>
  static accelerator_inline
  void coalescedWrite(T & c, const V & v) {
    ::coalescedWrite(c, v);
  }

  static constexpr ViewMode AcceleratorWriteMode = AcceleratorWriteDiscard;
  
};

template<typename T1, typename T2, typename Accumulator, int unary_expr, class Enable = void>
struct MultiplicationTable {};

// General
template<typename T1, typename vtype, typename Accumulator, int unary_expr>
struct MultiplicationTable<T1,iSinglet<vtype>,Accumulator,unary_expr> {
  typedef T1 result_type;
  typedef iSinglet<vtype> T2;
  static constexpr int n_elements = GridTypeMapper<T1>::count;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    Accumulator::coalescedWriteElement(c, coalescedReadElement(a, e) * coalescedRead(b()()()), e);
  }
};

// MSinglet x MSinglet
template<int n, typename vtype, typename Accumulator>
struct MultiplicationTable<iMatrix<iSinglet<vtype>,n>, iMatrix<iSinglet<vtype>,n>, Accumulator, BIT_COLORTRACE> {
  typedef iMatrix<iSinglet<vtype>,n> T1;
  typedef T1 T2;
  typedef iSinglet<vtype> result_type;
  static constexpr int n_elements = 1;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    decltype(coalescedRead(a(0,0))) v = Zero();
    for (int i=0;i<n;i++)
      for (int j=0;j<n;j++)
	v += coalescedRead(a(i,j)) * coalescedRead(b(j,i));
    Accumulator::coalescedWrite(c, v);
  }
};

template<int n, typename vtype, typename Accumulator>
struct MultiplicationTable<iMatrix<iSinglet<vtype>,n>, iMatrix<iSinglet<vtype>,n>, Accumulator, BIT_COLORTRACE|BIT_SPINTRACE> :
  MultiplicationTable<iMatrix<iSinglet<vtype>,n>, iMatrix<iSinglet<vtype>,n>, Accumulator, BIT_COLORTRACE> {};

template<int n, typename vtype, typename Accumulator, int unary_expr>
struct MultiplicationTable<iMatrix<iSinglet<vtype>,n>, iMatrix<iSinglet<vtype>,n>, Accumulator, unary_expr> {
  typedef iMatrix<iSinglet<vtype>,n> T1;
  typedef T1 T2;
  typedef T1 result_type;
  static constexpr int n_elements = n*n;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int i = e / n;
    int j = e % n;
    decltype(coalescedRead(a(0,0))) v = Zero();
    for (int l=0;l<n;l++)
      v += coalescedRead(a(i,l)) * coalescedRead(b(l,j));
    Accumulator::coalescedWrite(c(i,j), v);
  }
};


// MSinglet x VSinglet
template<int n, typename vtype, typename Accumulator, int unary_expr>
struct MultiplicationTable<iMatrix<iSinglet<vtype>,n>, iVector<iSinglet<vtype>,n>, Accumulator, unary_expr> {
  typedef iMatrix<iSinglet<vtype>,n> T1;
  typedef iVector<iSinglet<vtype>,n> T2;
  typedef T2 result_type;
  static constexpr int n_elements = n;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int i) {
    decltype(coalescedRead(b(0))) v = Zero();
    for (int j=0;j<n;j++)
      v += coalescedRead(a(i,j)) * coalescedRead(b(j));
    Accumulator::coalescedWrite(c(i), v);
  }
};

// MColor x MColor
template<int n, typename vtype, typename Accumulator, int unary_expr>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype,n>>>, iScalar<iScalar<iMatrix<vtype,n>>>, Accumulator, unary_expr> {
  typedef iScalar<iScalar<iMatrix<vtype,n>>> T1;
  typedef T1 T2;
  typedef T1 result_type;
  static constexpr int n_elements = n*n;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int i = e / n;
    int j = e % n;
    decltype(coalescedRead(a()()(0,0))) v = 0.0;
    for (int l=0;l<n;l++)
      v += coalescedRead(a()()(i,l)) * coalescedRead(b()()(l,j));
    Accumulator::coalescedWrite(c()()(i,j), v);
  }
};

template<int n, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype,n>>>, iScalar<iScalar<iMatrix<vtype,n>>>, Accumulator, BIT_COLORTRACE> {
  typedef iScalar<iScalar<iMatrix<vtype,n>>> T1;
  typedef T1 T2;
  typedef iSinglet<vtype> result_type;
  static constexpr int n_elements = 1;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    decltype(coalescedRead(a()()(0,0))) v = 0.0;
    for (int i=0;i<n;i++)
      for (int j=0;j<n;j++)
	v += coalescedRead(a()()(i,j)) * coalescedRead(b()()(j,i));
    Accumulator::coalescedWrite(c()()(), v);
  }
};

template<int n, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype,n>>>, iScalar<iScalar<iMatrix<vtype,n>>>, Accumulator, BIT_COLORTRACE|BIT_SPINTRACE> :
  MultiplicationTable<iScalar<iScalar<iMatrix<vtype,n>>>, iScalar<iScalar<iMatrix<vtype,n>>>, Accumulator, BIT_COLORTRACE> {};

// MColor x VColor
template<int n, typename vtype, typename Accumulator, int unary_expr>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype,n>>>, iScalar<iScalar<iVector<vtype,n>>>, Accumulator, unary_expr> {
  typedef iScalar<iScalar<iMatrix<vtype,n>>> T1;
  typedef iScalar<iScalar<iVector<vtype,n>>> T2;
  typedef T2 result_type;
  static constexpr int n_elements = n;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int i) {
    decltype(coalescedRead(b()()(0))) v = 0.0;
    for (int j=0;j<n;j++)
      v += coalescedRead(a()()(i,j)) * coalescedRead(b()()(j));
    Accumulator::coalescedWrite(c()()(i), v);
  }
};

// MSpin x MSpin
template<int n, typename vtype, typename Accumulator, int unary_expr>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype>,n>>, iScalar<iMatrix<iScalar<vtype>,n>>, Accumulator, unary_expr> {
  typedef iScalar<iMatrix<iScalar<vtype>,n>> T1;
  typedef T1 T2;
  typedef T1 result_type;
  static constexpr int n_elements = n*n;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int i = e / n;
    int j = e % n;
    decltype(coalescedRead(a()(0,0)())) v = 0.0;
    for (int l=0;l<n;l++)
      v += coalescedRead(a()(i,l)()) * coalescedRead(b()(l,j)());
    Accumulator::coalescedWrite(c()(i,j)(), v);
  }
};

template<int n, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype>,n>>, iScalar<iMatrix<iScalar<vtype>,n>>, Accumulator, BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iScalar<vtype>,n>> T1;
  typedef T1 T2;
  typedef iSinglet<vtype> result_type;
  static constexpr int n_elements = 1;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    decltype(coalescedRead(a()(0,0)())) v = 0.0;
    for (int i=0;i<n;i++)
      for (int j=0;j<n;j++)
	v += coalescedRead(a()(i,j)()) * coalescedRead(b()(j,i)());
    Accumulator::coalescedWrite(c()()(), v);
  }
};

template<int n, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype>,n>>, iScalar<iMatrix<iScalar<vtype>,n>>, Accumulator, BIT_SPINTRACE|BIT_COLORTRACE> :
  MultiplicationTable<iScalar<iMatrix<iScalar<vtype>,n>>, iScalar<iMatrix<iScalar<vtype>,n>>, Accumulator, BIT_SPINTRACE> {};


// MSpin x VSpin
template<int n, typename vtype, typename Accumulator, int unary_expr>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype>,n>>, iScalar<iVector<iScalar<vtype>,n>>, Accumulator, unary_expr> {
  typedef iScalar<iMatrix<iScalar<vtype>,n>> T1;
  typedef iScalar<iVector<iScalar<vtype>,n>> T2;
  typedef T2 result_type;
  static constexpr int n_elements = n;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int i) {
    decltype(coalescedRead(b()(0)())) v = 0.0;
    for (int j=0;j<n;j++)
      v += coalescedRead(a()(i,j)()) * coalescedRead(b()(j)());
    Accumulator::coalescedWrite(c()(i)(), v);
  }
};

// MColor x VSpinColor
template<int ns, int nc, typename vtype, typename Accumulator, int unary_expr>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype,nc>>>, iScalar<iVector<iVector<vtype,nc>,ns>>, Accumulator, unary_expr> {
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> T1;
  typedef iScalar<iVector<iVector<vtype,nc>,ns>> T2;
  typedef T2 result_type;
  static constexpr int n_elements = nc*ns;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int is = e / nc;
    int ic = e % nc;
    decltype(coalescedRead(b()(0)(0))) v = 0.0;
    for (int jc=0;jc<nc;jc++)
      v += coalescedRead(a()()(ic,jc)) * coalescedRead(b()(is)(jc));
    Accumulator::coalescedWrite(c()(is)(ic), v);
  }
};

// MColor x MSpinColor
template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype,nc>>>, iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, Accumulator, 0> {
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T2;
  typedef T2 result_type;
  static constexpr int n_elements = nc*ns*nc*ns;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int ie = e % (nc * ns);
    int je = e / (nc * ns);
      
    int is = ie / nc;
    int ic = ie % nc;

    int js = je / nc;
    int jc = je % nc;
    
    decltype(coalescedRead(b()(0,0)(0,0))) v = 0.0;
    for (int lc=0;lc<nc;lc++)
      v += coalescedRead(a()()(ic,lc)) * coalescedRead(b()(is,js)(lc,jc));
    Accumulator::coalescedWrite(c()(is,js)(ic,jc), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype,nc>>>, iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, Accumulator, BIT_COLORTRACE> {
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T2;
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> result_type;
  static constexpr int n_elements = ns*ns;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int is = e / ns;
    int js = e % ns;
    
    decltype(coalescedRead(b()(0,0)(0,0))) v = 0.0;
    for (int ic=0;ic<nc;ic++)
      for (int lc=0;lc<nc;lc++)
	v += coalescedRead(a()()(ic,lc)) * coalescedRead(b()(is,js)(lc,ic));
    Accumulator::coalescedWrite(c()(is,js)(), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype,nc>>>, iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, Accumulator, BIT_SPINTRACE> {
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T2;
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> result_type;
  static constexpr int n_elements = nc*nc;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int ic = e % nc;
    int jc = e / nc;
    
    decltype(coalescedRead(b()(0,0)(0,0))) v = 0.0;
    for (int ls=0;ls<ns;ls++)
      for (int lc=0;lc<nc;lc++)
	v += coalescedRead(a()()(ic,lc)) * coalescedRead(b()(ls,ls)(lc,jc));
    Accumulator::coalescedWrite(c()()(ic,jc), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iScalar<iMatrix<vtype,nc>>>, iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, Accumulator, BIT_SPINTRACE|BIT_COLORTRACE> {
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T2;
  typedef iSinglet<vtype> result_type;
  static constexpr int n_elements = 1;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    decltype(coalescedRead(b()(0,0)(0,0))) v = 0.0;
    for (int ic=0;ic<nc;ic++)
      for (int ls=0;ls<ns;ls++)
	for (int lc=0;lc<nc;lc++)
	  v += coalescedRead(a()()(ic,lc)) * coalescedRead(b()(ls,ls)(lc,ic));
    Accumulator::coalescedWrite(c()()(), v);
  }
};

// MSpinColor x MColor
template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, iScalar<iScalar<iMatrix<vtype,nc>>>, Accumulator, 0> {
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T1;
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> T2;
  typedef T1 result_type;
  static constexpr int n_elements = nc*ns*nc*ns;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int ie = e % (nc * ns);
    int je = e / (nc * ns);
      
    int is = ie / nc;
    int ic = ie % nc;

    int js = je / nc;
    int jc = je % nc;
    
    decltype(coalescedRead(b()()(0,0))) v = 0.0;
    for (int lc=0;lc<nc;lc++)
      v += coalescedRead(a()(is,js)(ic,lc)) * coalescedRead(b()()(lc,jc));
    Accumulator::coalescedWrite(c()(is,js)(ic,jc), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, iScalar<iScalar<iMatrix<vtype,nc>>>, Accumulator, BIT_COLORTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T1;
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> T2;
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> result_type;
  static constexpr int n_elements = ns*ns;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int is = e / ns;
    int js = e % ns;
    
    decltype(coalescedRead(b()()(0,0))) v = 0.0;
    for (int lc=0;lc<nc;lc++)
      for (int ic=0;ic<nc;ic++)
	v += coalescedRead(a()(is,js)(ic,lc)) * coalescedRead(b()()(lc,ic));
    Accumulator::coalescedWrite(c()(is,js)(), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, iScalar<iScalar<iMatrix<vtype,nc>>>, Accumulator, BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T1;
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> T2;
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> result_type;
  static constexpr int n_elements = nc*nc;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int ic = e / nc;
    int jc = e % nc;
    
    decltype(coalescedRead(b()()(0,0))) v = 0.0;
    for (int lc=0;lc<nc;lc++)
      for (int is=0;is<ns;is++)
	v += coalescedRead(a()(is,is)(ic,lc)) * coalescedRead(b()()(lc,jc));
    Accumulator::coalescedWrite(c()()(ic,jc), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, iScalar<iScalar<iMatrix<vtype,nc>>>, Accumulator, BIT_SPINTRACE|BIT_COLORTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T1;
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> T2;
  typedef iSinglet<vtype> result_type;
  static constexpr int n_elements = 1;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    
    decltype(coalescedRead(b()()(0,0))) v = 0.0;
    for (int lc=0;lc<nc;lc++)
      for (int is=0;is<ns;is++)
	for (int ic=0;ic<nc;ic++)
	  v += coalescedRead(a()(is,is)(ic,lc)) * coalescedRead(b()()(lc,ic));
    Accumulator::coalescedWrite(c()()(), v);
  }
};


// MSpin x VSpinColor
template<int ns, int nc, typename vtype, typename Accumulator, int unary_expr>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype>,ns>>, iScalar<iVector<iVector<vtype,nc>,ns>>, Accumulator, unary_expr> {
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> T1;
  typedef iScalar<iVector<iVector<vtype,nc>,ns>> T2;
  typedef T2 result_type;
  static constexpr int n_elements = nc*ns;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int is = e / nc;
    int ic = e % nc;
    decltype(coalescedRead(b()(0)(0))) v = 0.0;
    for (int js=0;js<ns;js++)
      v += coalescedRead(a()(is,js)()) * coalescedRead(b()(js)(ic));
    Accumulator::coalescedWrite(c()(is)(ic), v);
  }
};

// MSpin x MSpinColor
template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype>,ns>>, iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, Accumulator, 0> {
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T2;
  typedef T2 result_type;
  static constexpr int n_elements = nc*ns*nc*ns;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int ie = e % (nc * ns);
    int je = e / (nc * ns);
      
    int is = ie / nc;
    int ic = ie % nc;

    int js = je / nc;
    int jc = je % nc;
    
    decltype(coalescedRead(b()(0,0)(0,0))) v = 0.0;
    for (int ls=0;ls<ns;ls++)
      v += coalescedRead(a()(is,ls)()) * coalescedRead(b()(ls,js)(ic,jc));
    Accumulator::coalescedWrite(c()(is,js)(ic,jc), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype>,ns>>, iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, Accumulator, BIT_COLORTRACE> {
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T2;
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> result_type;
  static constexpr int n_elements = ns*ns;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int is = e / ns;
    int js = e % ns;
    
    decltype(coalescedRead(b()(0,0)(0,0))) v = 0.0;
    for (int ic=0;ic<nc;ic++)
      for (int ls=0;ls<ns;ls++)
	v += coalescedRead(a()(is,ls)()) * coalescedRead(b()(ls,js)(ic,ic));
    Accumulator::coalescedWrite(c()(is,js)(), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype>,ns>>, iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, Accumulator, BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T2;
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> result_type;
  static constexpr int n_elements = nc*nc;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int ic = e / nc;
    int jc = e % nc;
    
    decltype(coalescedRead(b()(0,0)(0,0))) v = 0.0;
    for (int ls=0;ls<ns;ls++)
      for (int is=0;is<ns;is++)
      v += coalescedRead(a()(is,ls)()) * coalescedRead(b()(ls,is)(ic,jc));
    Accumulator::coalescedWrite(c()()(ic,jc), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iScalar<vtype>,ns>>, iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, Accumulator, BIT_SPINTRACE|BIT_COLORTRACE> {
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T2;
  typedef iSinglet<vtype> result_type;
  static constexpr int n_elements = 1;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    
    decltype(coalescedRead(b()(0,0)(0,0))) v = 0.0;
    for (int ls=0;ls<ns;ls++)
      for (int is=0;is<ns;is++)
	for (int ic=0;ic<nc;ic++)
	  v += coalescedRead(a()(is,ls)()) * coalescedRead(b()(ls,is)(ic,ic));
    Accumulator::coalescedWrite(c()()(), v);
  }
};



// MSpinColor x MSpin
template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, iScalar<iMatrix<iScalar<vtype>,ns>>, Accumulator, 0> {
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T1;
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> T2;
  typedef T1 result_type;
  static constexpr int n_elements = nc*ns*nc*ns;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int ie = e % (nc * ns);
    int je = e / (nc * ns);
      
    int is = ie / nc;
    int ic = ie % nc;

    int js = je / nc;
    int jc = je % nc;
    
    decltype(coalescedRead(b()(0,0)())) v = 0.0;
    for (int ls=0;ls<ns;ls++)
      v += coalescedRead(a()(is,ls)(ic,jc)) * coalescedRead(b()(ls,js)());
    Accumulator::coalescedWrite(c()(is,js)(ic,jc), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, iScalar<iMatrix<iScalar<vtype>,ns>>, Accumulator, BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T1;
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> T2;
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> result_type;
  static constexpr int n_elements = nc*nc;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int ic = e / nc;
    int jc = e % nc;
    
    decltype(coalescedRead(b()(0,0)())) v = 0.0;
    for (int ls=0;ls<ns;ls++)
      for (int is=0;is<ns;is++)
	v += coalescedRead(a()(is,ls)(ic,jc)) * coalescedRead(b()(ls,is)());
    Accumulator::coalescedWrite(c()()(ic,jc), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, iScalar<iMatrix<iScalar<vtype>,ns>>, Accumulator, BIT_COLORTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T1;
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> T2;
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> result_type;
  static constexpr int n_elements = ns*ns;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int is = e / ns;
    int js = e % ns;
    
    decltype(coalescedRead(b()(0,0)())) v = 0.0;
    for (int ls=0;ls<ns;ls++)
      for (int ic=0;ic<nc;ic++)
	v += coalescedRead(a()(is,ls)(ic,ic)) * coalescedRead(b()(ls,js)());
    Accumulator::coalescedWrite(c()(is,js)(), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, iScalar<iMatrix<iScalar<vtype>,ns>>, Accumulator, BIT_COLORTRACE|BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T1;
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> T2;
  typedef iSinglet<vtype> result_type;
  static constexpr int n_elements = 1;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    decltype(coalescedRead(b()(0,0)())) v = 0.0;
    for (int ls=0;ls<ns;ls++)
      for (int ic=0;ic<nc;ic++)
	for (int is=0;is<ns;is++)
	  v += coalescedRead(a()(is,ls)(ic,ic)) * coalescedRead(b()(ls,is)());
    Accumulator::coalescedWrite(c()()(), v);
  }
};

// MSpinColor x MSpinColor
template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, Accumulator, 0> {
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T2;
  typedef T2 result_type;
  static constexpr int n_elements = nc*ns*nc*ns;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int ie = e % (nc * ns);
    int je = e / (nc * ns);
      
    int is = ie / nc;
    int ic = ie % nc;

    int js = je / nc;
    int jc = je % nc;
    
    decltype(coalescedRead(b()(0,0)(0,0))) v = 0.0;
    for (int ls=0;ls<ns;ls++)
      for (int lc=0;lc<nc;lc++)
	v += coalescedRead(a()(is,ls)(ic,lc)) * coalescedRead(b()(ls,js)(lc,jc));
    Accumulator::coalescedWrite(c()(is,js)(ic,jc), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, Accumulator, BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T2;
  typedef iScalar<iScalar<iMatrix<vtype,nc>>> result_type;
  static constexpr int n_elements = nc*nc;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int ic = e / nc;
    int jc = e % nc;
    
    decltype(coalescedRead(b()(0,0)(0,0))) v = 0.0;
    for (int ls=0;ls<ns;ls++)
      for (int lc=0;lc<nc;lc++)
	for (int is=0;is<ns;is++)
	v += coalescedRead(a()(is,ls)(ic,lc)) * coalescedRead(b()(ls,is)(lc,jc));
    Accumulator::coalescedWrite(c()()(ic,jc), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, Accumulator, BIT_COLORTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T2;
  typedef iScalar<iMatrix<iScalar<vtype>,ns>> result_type;
  static constexpr int n_elements = ns*ns;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int is = e / ns;
    int js = e % ns;
    
    decltype(coalescedRead(b()(0,0)(0,0))) v = 0.0;
    for (int ls=0;ls<ns;ls++)
      for (int lc=0;lc<nc;lc++)
	for (int ic=0;ic<nc;ic++)
	  v += coalescedRead(a()(is,ls)(ic,lc)) * coalescedRead(b()(ls,js)(lc,ic));
    Accumulator::coalescedWrite(c()(is,js)(), v);
  }
};

template<int ns, int nc, typename vtype, typename Accumulator>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, Accumulator, BIT_COLORTRACE|BIT_SPINTRACE> {
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T1;
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T2;
  typedef iSinglet<vtype> result_type;
  static constexpr int n_elements = 1;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    
    decltype(coalescedRead(b()(0,0)(0,0))) v = 0.0;
    for (int ls=0;ls<ns;ls++)
      for (int lc=0;lc<nc;lc++)
	for (int ic=0;ic<nc;ic++)
	  for (int is=0;is<ns;is++)
	    v += coalescedRead(a()(is,ls)(ic,lc)) * coalescedRead(b()(ls,is)(lc,ic));
    Accumulator::coalescedWrite(c()()(), v);
  }
};

// MSpinColor x VSpinColor
template<int ns, int nc, typename vtype, typename Accumulator, int unary_expr>
struct MultiplicationTable<iScalar<iMatrix<iMatrix<vtype,nc>,ns>>, iScalar<iVector<iVector<vtype,nc>,ns>>, Accumulator, unary_expr> {
  typedef iScalar<iMatrix<iMatrix<vtype,nc>,ns>> T1;
  typedef iScalar<iVector<iVector<vtype,nc>,ns>> T2;
  typedef T2 result_type;
  static constexpr int n_elements = nc*ns;
  static accelerator_inline void eval(result_type & c, const T1 & a, const T2 & b, int e) {
    int is = e / nc;
    int ic = e % nc;
    
    decltype(coalescedRead(b()(0)(0))) v = 0.0;
    for (int js=0;js<ns;js++)
      for (int jc=0;jc<nc;jc++)
	v += coalescedRead(a()(is,js)(ic,jc)) * coalescedRead(b()(js)(jc));
    Accumulator::coalescedWrite(c()(is)(ic), v);
  }
};

template<typename T1, typename T2, typename Accumulator, int unary_expr>
cgpt_Lattice_base* cgpt_mul_acc_unary(cgpt_Lattice_base* _c,
				const Lattice<T1> & a,
				const Lattice<T2> & b) {
    
  GridBase* grid = a.Grid();

  typedef MultiplicationTable<T1,T2,Accumulator,unary_expr> MT;
  typedef typename MT::result_type T;

  if (!_c)
    _c = new cgpt_Lattice<T>(grid);

  auto & c = compatible<T>(_c)->l;
  c.Checkerboard() = a.Checkerboard();

  autoView(c_v, c, Accumulator::AcceleratorWriteMode);
  autoView(a_v, a, AcceleratorRead);
  autoView(b_v, b, AcceleratorRead);

  auto * p_c = &c_v[0];
  auto * p_a = &a_v[0];
  auto * p_b = &b_v[0];

  accelerator_for2d(osite, grid->oSites(), j, MT::n_elements, grid->Nsimd(), {
      MT::eval(p_c[osite], p_a[osite], p_b[osite], j);
    });

  return _c;
}

template<typename T1, typename T2, typename Accumulator>
cgpt_Lattice_base* cgpt_mul_acc(cgpt_Lattice_base* _c,
				const Lattice<T1> & a,
				const Lattice<T2> & b,
				int unary_expr) {

  switch (unary_expr) {
  case 0:
    return cgpt_mul_acc_unary<T1,T2,Accumulator,0>(_c,a,b);
  case BIT_SPINTRACE:
    return cgpt_mul_acc_unary<T1,T2,Accumulator,BIT_SPINTRACE>(_c,a,b);
  case BIT_COLORTRACE:
    return cgpt_mul_acc_unary<T1,T2,Accumulator,BIT_COLORTRACE>(_c,a,b);
  case BIT_COLORTRACE|BIT_SPINTRACE:
    return cgpt_mul_acc_unary<T1,T2,Accumulator,BIT_SPINTRACE|BIT_COLORTRACE>(_c,a,b);
  default:
    ERR("Unknown unary %d", unary_expr);
  }
}

template<typename T1, typename T2>
cgpt_Lattice_base* cgpt_mul(cgpt_Lattice_base* _c, bool ac,
			    const Lattice<T1> & a,
			    const Lattice<T2> & b,
			    int unary_expr) {

  if (ac) {
    ASSERT(_c);
    return cgpt_mul_acc<T1,T2,AccumulatorYes>(_c,a,b,unary_expr);
  } else {
    return cgpt_mul_acc<T1,T2,AccumulatorNo>(_c,a,b,unary_expr);
  }
}

template<typename A, typename B>
  cgpt_Lattice_base* lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, const A& la, int unary_b, const B& lb,int unary_expr) {
  ASSERT(la.Grid() == lb.Grid());
  return cgpt_mul(dst, ac, cgpt_unary(unary_a, la), cgpt_unary(unary_b, lb), unary_expr);
}

#define _COMPATIBLE_(t) typeOpen(b,t) { return lattice_mul(dst,ac, unary_a,la,unary_b,lb,unary_expr); } typeClose();
#define _COMPATIBLE_MSR_(t) typeOpen(b,t) { return lattice_mul(dst,ac, unary_a,la,unary_b,lb,unary_expr); } typeClose();

#define _OUTER_PRODUCT_(t) if (unary_a == 0 && unary_b == (BIT_TRANS|BIT_CONJ)) { typeOpen(b,t) { return lattice_unary_lat(dst,ac, outerProduct(la,lb), unary_expr); } typeClose(); }
#define _INNER_PRODUCT_(t) if (unary_a == (BIT_TRANS|BIT_CONJ) && unary_b == 0) { typeOpen(b,t) { return lattice_unary_lat(dst, ac, localInnerProduct(la,lb), unary_expr ); } typeClose(); }

