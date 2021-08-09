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


    Some compilers do not properly align the memory of the simd types.

    Cannot map this to Grid::Vector because we need thread safety of
    allocators.
*/
void* alignedAlloc(size_t size) {
  return cgpt_alloc(sizeof(vInteger),size);
}

void alignedFree(void* p) {
  free(p);
}

template<class T>
class ThreadSafeAlignedVector {
  T* p;
  size_t n;
 public:
  ThreadSafeAlignedVector(size_t _n) : n(_n) {
    p = (T*)alignedAlloc(n * sizeof(T));
    assert(p);
  }

  ThreadSafeAlignedVector(const ThreadSafeAlignedVector& other) : n(other.n) {
    p = (T*)alignedAlloc(n * sizeof(T));
    assert(p);
    memcpy(p,other.p,sizeof(T) * n);
  }

  ~ThreadSafeAlignedVector() {
    alignedFree(p);
  }

  ThreadSafeAlignedVector& operator=(const ThreadSafeAlignedVector& other) {
    alignedFree(p);
    n = other.n;
    p = (T*)alignedAlloc(n * sizeof(T));
    memcpy(p,other.p,sizeof(T) * n);
    return *this;
  }

  T & operator[](size_t i) {
    assert(i < n);
    return p[i];
  }

  const T & operator[](size_t i) const {
    assert(i < n);
    return p[i];
  }

  size_t size() const {
    return n;
  }
};
