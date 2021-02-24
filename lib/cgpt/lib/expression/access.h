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
