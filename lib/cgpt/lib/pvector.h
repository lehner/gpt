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
class PVector {
 protected:
  std::vector<T*> _v;

 public:
  PVector(long size) : _v(size) {
  }

  PVector() : _v() {
  }

  void resize(long size) {
    _v.resize(size);
  }

  void push_back(T* t) {
    _v.push_back(t);
  }

  long size() const {
    return _v.size();
  }

  T& operator[](long i) {
    return *_v[i];
  }

  const T& operator[](long i) const {
    return *_v[i];
  }

  T*& operator()(long i) {
    return _v[i];
  }

  const T*& operator()(long i) const {
    return _v[i];
  }

  // allow for type conversion to access first element
  operator T&() {
    return *_v[0];
  }

  PVector slice(long i0, long i1, long step = 1) const {
    if (i0<0)
      i0=0;
    if (i1 > _v.size())
      i1 = _v.size();
    PVector ret;
    for (long i=i0;i<i1;i+=step)
      ret.push_back(_v[i]);
    return ret;
  }
};

template<typename T>
class PMatrix {
 protected:
  T* _v;
  int _N;
 public:
  PMatrix(T* v, int N) : _v(v), _N(N) {
  }

  T& operator()(int i, int j) {
    return _v[i*_N + j];
  }

  const T& operator()(int i, int j) const {
    return _v[i*_N + j];
  }
};
