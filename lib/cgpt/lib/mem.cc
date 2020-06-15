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
#include "lib.h"

//#define MEM_DEBUG 1
void* operator new(size_t size) {
  // makes sure SIMD types are properly aligned
  // negligible overhead for other data
  void* r = aligned_alloc(sizeof(vInteger),size);
  if (!r) // may happen on some implementations for size < sizeof(vInteger)
    r = malloc(size);
#ifdef MEM_DEBUG
  printf("Alloc %p\n",r);
#endif
  return r;
}

void operator delete(void* p) {
#ifdef MEM_DEBUG  
  printf("Delete %p\n",p);
#endif
  free(p);
}
