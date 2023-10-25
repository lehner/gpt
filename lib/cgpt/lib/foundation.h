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

    
    This file tries to isolate foundational code of the data parallel layer.
    (Some of this could move to Grid.)
*/
#include <Grid/Grid.h>

using namespace Grid;

#if defined(GRID_SYCL) || defined(GRID_CUDA) || defined(GRID_HIP)
#define GRID_HAS_ACCELERATOR
#endif

#if defined (GRID_COMMS_MPI3)
#define CGPT_USE_MPI 1
#endif


#define VECTOR_VIEW_OPEN(l,v,mode)					\
  Vector< decltype(l[0].View(mode)) > __ ## v; __ ## v.reserve(l.size()); \
  Vector< decltype(&l[0].View(mode)[0]) > _ ## v; _ ## v.resize(l.size()); \
  for(uint64_t k=0;k<l.size();k++) {					\
    __ ## v.push_back(l[k].View(mode));					\
    _ ## v[k] = &__ ## v[k][0];						\
  }									\
  auto v = & _ ## v[0];

#define VECTOR_VIEW_CLOSE(v)						\
  for(uint64_t k=0;k<__ ## v.size();k++) __ ## v[k].ViewClose();


#define VECTOR_ELEMENT_VIEW_OPEN(ET, l, v, mode)			\
  Vector<ET*> _ ## v; _ ## v.reserve(l.size());				\
  Vector<int> __ ## v; __ ## v.reserve(l.size());			\
  for(uint64_t k=0;k<l.size();k++) {					\
    _ ## v.push_back((ET*)l[k]->memory_view_open(mode));			\
    long Nsimd, word, simd_word;					\
    l[k]->describe_data_layout(Nsimd,word,simd_word);			\
    __ ## v.push_back(word / simd_word);				\
  }									\
  auto v = & _ ## v[0];							\
  auto v ## _nelements = & __ ## v[0];

#define VECTOR_ELEMENT_VIEW_CLOSE(l)					\
  for(uint64_t k=0;k<l.size();k++) l[k]->memory_view_close();



NAMESPACE_BEGIN(Grid);

// aligned vector
template<class T> using AlignedVector = std::vector<T,alignedAllocator<T> >;

#include "foundation/access.h"
#include "foundation/reduce.h"
#include "foundation/unary.h"
#include "foundation/binary.h"
#include "foundation/ternary.h"
#include "foundation/et.h"
#include "foundation/grid.h"
#include "foundation/block_lookup_table.h"
#include "foundation/block_core.h"
#include "foundation/transfer.h"
#include "foundation/basis.h"
#include "foundation/eigen.h"
#include "foundation/matrix.h"
#include "foundation/coarse.h"
#include "foundation/transform.h"


NAMESPACE_END(Grid);
