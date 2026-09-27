/*
    GPT - Grid Python Toolkit
    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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
#pragma once

// flat element index of the transpose (every iMatrix level transposed, as
// cgpt_trans): element e of trans(x) is element map(e) of x
template<typename T>
struct cgpt_transpose_index {
  static accelerator_inline int map(int e) { return e; }
};

template<typename T>
struct cgpt_transpose_index<iScalar<T>> {
  static accelerator_inline int map(int e) { return cgpt_transpose_index<T>::map(e); }
};

template<typename T, int n>
struct cgpt_transpose_index<iVector<T,n>> {
  static accelerator_inline int map(int e) {
    constexpr int c = GridTypeMapper<T>::count;
    return (e / c) * c + cgpt_transpose_index<T>::map(e % c);
  }
};

template<typename T, int n>
struct cgpt_transpose_index<iMatrix<T,n>> {
  static accelerator_inline int map(int e) {
    constexpr int c = GridTypeMapper<T>::count;
    int k = e / c;
    return ((k % n) * n + k / n) * c + cgpt_transpose_index<T>::map(e % c);
  }
};

// an operand read with a factor unary (BIT_TRANS|BIT_CONJ) applied on the
// fly, so that kernels need no temporary for unary(x)
template<typename T>
struct cgpt_unary_ref {
  const T & x;
  int unary;
};

template<typename T>
accelerator_inline
auto coalescedReadElement(const cgpt_unary_ref<T> & r, int e) -> decltype(coalescedReadElement(r.x, 0)) {
  auto v = coalescedReadElement(r.x, (r.unary & BIT_TRANS) ? cgpt_transpose_index<T>::map(e) : e);
  if (r.unary & BIT_CONJ)
    v = conjugate(v);
  return v;
}

// multi-index reads: a transpose swaps the (row, column) index pair at each
// matrix level, so the transposed element is addressed directly (no flat
// index map in inner loops)
template<typename T, int n>
accelerator_inline int cgpt_element_index_transposed(const iMatrix<T,n> & c, int a, int b) {
  return cgpt_element_index(c, b, a);
}

template<typename T, int n1, int n2>
accelerator_inline int cgpt_element_index_transposed(const iVector<iVector<T,n2>,n1> & c, int a, int b) {
  return cgpt_element_index(c, a, b);
}

template<typename T, int n1, int n2>
accelerator_inline int cgpt_element_index_transposed(const iMatrix<iMatrix<T,n2>,n1> & c, int a1, int b1, int a2, int b2) {
  return cgpt_element_index(c, b1, a1, b2, a2);
}

template<typename T, typename... I>
accelerator_inline int cgpt_element_index_transposed(const iScalar<T> & c, I... idx) {
  return cgpt_element_index_transposed(c(), idx...);
}

template<typename T, typename... I>
accelerator_inline
auto cgpt_unary_read(const cgpt_unary_ref<T> & r, I... idx) -> decltype(coalescedReadElement(r.x, 0)) {
  auto v = coalescedReadElement(r.x, (r.unary & BIT_TRANS) ? cgpt_element_index_transposed(r.x, idx...) : cgpt_element_index(r.x, idx...));
  if (r.unary & BIT_CONJ)
    v = conjugate(v);
  return v;
}

template<typename T>
accelerator_inline
auto coalescedReadElement(const cgpt_unary_ref<T> & r, int a, int b) -> decltype(coalescedReadElement(r.x, 0)) {
  return cgpt_unary_read(r, a, b);
}

template<typename T>
accelerator_inline
auto coalescedReadElement(const cgpt_unary_ref<T> & r, int a1, int b1, int a2, int b2) -> decltype(coalescedReadElement(r.x, 0)) {
  return cgpt_unary_read(r, a1, b1, a2, b2);
}

template<typename T>
accelerator_inline
auto coalescedRead(const cgpt_unary_ref<T> & r) -> decltype(coalescedRead(r.x)) {
  typedef decltype(coalescedRead(r.x)) O;
  typedef typename O::vector_type E;
  constexpr int n = sizeof(O) / sizeof(E);
  O v = coalescedRead(r.x), w;
  const E* pv = (const E*)&v;
  E* pw = (E*)&w;
  for (int e=0;e<n;e++) {
    E y = pv[(r.unary & BIT_TRANS) ? cgpt_transpose_index<O>::map(e) : e];
    pw[e] = (r.unary & BIT_CONJ) ? conjugate(y) : y;
  }
  return w;
}
