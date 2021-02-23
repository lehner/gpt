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

#ifndef GRID_SIMT

template<typename T>
accelerator_inline
void coalescedWriteElement(T & __restrict__ c, const typename T::vector_type & v, int e) {
  typedef typename T::vector_type vCoeff_t;
  vCoeff_t * __restrict__ p = (vCoeff_t*)&c;
  p[e] = v;
}

template<typename T>
accelerator_inline
typename T::vector_type coalescedReadElement(const T & __restrict__ c, int e) {
  typedef typename T::vector_type vCoeff_t;
  vCoeff_t * __restrict__ p = (vCoeff_t*)&c;
  return p[e];
}

#else

template<int Nsimd>
struct coalescedElementSIMT {
  
  template<typename T>
  static accelerator_inline
  void write(T & __restrict__ c, const typename T::scalar_type & v, int e) {
    typedef typename T::scalar_type Coeff_t;
    int lane=acceleratorSIMTlane(Nsimd);
    Coeff_t * __restrict__ p = (Coeff_t*)&c;
    p[e * Nsimd + lane] = v;
  }
  
  template<typename T>
  static accelerator_inline
  typename T::scalar_type read(const T & __restrict__ c, int e) {
    typedef typename T::scalar_type Coeff_t;
    int lane=acceleratorSIMTlane(Nsimd);
    Coeff_t * __restrict__ p = (Coeff_t*)&c;
    return p[e * Nsimd + lane];
  }
};
  
template<>
struct coalescedElementSIMT<1> {

  template<typename T>
  static accelerator_inline
  void write(T & __restrict__ c, const typename T::scalar_type & v, int e) {
    typedef typename T::scalar_type Coeff_t;
    Coeff_t * __restrict__ p = (Coeff_t*)&c;
    p[e] = v;
  }
  
  template<typename T>
  static accelerator_inline
  typename T::scalar_type read(const T & __restrict__ c, int e) {
    typedef typename T::scalar_type Coeff_t;
    Coeff_t * __restrict__ p = (Coeff_t*)&c;
    return p[e];
  }
};
  

template<typename T>
accelerator_inline
void coalescedWriteElement(T & __restrict__ c, const typename T::scalar_type & v, int e) {
  coalescedElementSIMT<T::Nsimd()>::write(c,v,e);
}

template<typename T>
accelerator_inline
typename T::scalar_type coalescedReadElement(const T & __restrict__ c, int e) {
  return coalescedElementSIMT<T::Nsimd()>::read(c,e);
}

#endif


template<typename T, int n>
accelerator_inline
auto coalescedReadElement(const iMatrix<T,n> & c, int a, int b) -> decltype(coalescedReadElement(c,0)) {
  return coalescedReadElement(c,a*n + b);
}

template<typename T, int n1, int n2>
accelerator_inline
auto coalescedReadElement(const iVector<iVector<T,n2>,n1> & c, int a, int b) -> decltype(coalescedReadElement(c,0)) {
  return coalescedReadElement(c,a*n2 + b);
}

template<typename T, int n1, int n2>
accelerator_inline
auto coalescedReadElement(const iMatrix<iMatrix<T,n2>,n1> & c, int a1, int b1, int a2, int b2) -> decltype(coalescedReadElement(c,0)) {
  return coalescedReadElement(c,a2*n2 + b2 + n2*n2*(a1*n1 + b1));
}

template<typename T>
accelerator_inline
auto coalescedReadElement(const iScalar<T> & c, int a, int b) -> decltype(coalescedReadElement(c(),0)) {
  return coalescedReadElement(c(),a,b);
}

template<typename T>
accelerator_inline
auto coalescedReadElement(const iScalar<T> & c, int a1, int b1, int a2, int b2) -> decltype(coalescedReadElement(c(),0)) {
  return coalescedReadElement(c(),a1,b1,a2,b2);
}


class AccumulatorYesBase {
public:
  
  template<typename T, typename V>
  static accelerator_inline
  void coalescedWriteElement(T & c, const V & v, int e) {
    auto r = coalescedReadElement(c, e);
    r += v;
    ::coalescedWriteElement(c, r, e);
  }

  template<typename T, typename V>
  static accelerator_inline
  void coalescedWrite(T & c, const V & v) {
    auto r = coalescedRead(c);
    r += v;
    ::coalescedWrite(c, r);
  }

  static constexpr ViewMode AcceleratorWriteMode = AcceleratorWrite;

};

class AccumulatorNoBase {
public:
  
  template<typename T, typename V>
  static accelerator_inline
  void coalescedWriteElement(T & c, const V & v, int e) {
    ::coalescedWriteElement(c, v, e);
  }

  template<typename T, typename V>
  static accelerator_inline
  void coalescedWrite(T & c, const V & v) {
    ::coalescedWrite(c, v);
  }

  static constexpr ViewMode AcceleratorWriteMode = AcceleratorWriteDiscard;
  
};

template<typename Base>
class AccumulatorImpl : public Base {
public:
  
  template<typename T, typename V>
  static accelerator_inline
  void coalescedWriteElement(T & c, const V & v, int e) {
    Base::coalescedWriteElement(c,v,e);
  }

  template<typename T, typename V>
  static accelerator_inline
  void coalescedWriteSinglet(T & c, const V & v) {
    Base::coalescedWriteElement(c,v,0);
  }

  template<typename T, typename V>
  static accelerator_inline
  void coalescedWriteElement(T & c, const iScalar<V> & v, int e) {
    coalescedWriteElement(c,v(),e);
  }


  template<typename T, typename V>
  static accelerator_inline
  void coalescedWrite(T & c, const V & v) {
    Base::coalescedWrite(c,v);
  }

  template<typename T, typename V, int n>
  static accelerator_inline
  void coalescedWrite(iMatrix<T,n> & c, int i, int j, const V & v) {
    coalescedWriteElement(c, v, i*n + j);
  }

  template<typename T, typename V, int n1, int n2>
  static accelerator_inline
  void coalescedWrite(iVector<iVector<T,n2>,n1> & c, int i, int j, const V & v) {
    coalescedWriteElement(c, v, i*n2 + j);
  }

  template<typename T, typename V, int n1, int n2>
  static accelerator_inline
  void coalescedWrite(iMatrix<iMatrix<T,n2>,n1> & c, int i1, int j1, int i2, int j2, const V & v) {
    coalescedWriteElement(c, v, i2*n2 + j2 + n2*n2*(i1*n1 + j1));
  }

  template<typename T, typename V, int n>
  static accelerator_inline
  void coalescedWrite(iVector<T,n> & c, int i, const V & v) {
    coalescedWriteElement(c, v, i);
  }

  template<typename T, typename V>
  static accelerator_inline
  void coalescedWrite(iScalar<T> & c, int i1, int j1, int i2, int j2, const V & v) {
    coalescedWrite(c(), i1, j1, i2, j2, v);
  }

  template<typename T, typename V>
  static accelerator_inline
  void coalescedWrite(iScalar<T> & c, int i, int j, const V & v) {
    coalescedWrite(c(), i, j, v);
  }

  template<typename T, typename V>
  static accelerator_inline
  void coalescedWrite(iScalar<T> & c, int i, const V & v) {
    coalescedWrite(c(), i, v);
  }


};

typedef AccumulatorImpl<AccumulatorYesBase> AccumulatorYes;
typedef AccumulatorImpl<AccumulatorNoBase> AccumulatorNo;
