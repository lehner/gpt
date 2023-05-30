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

#define BCOPY_MEM_ALIGN   (sizeof(vComplexF))
#define BCOPY_ALIGN(sz) ( ( ( (sz) + BCOPY_MEM_ALIGN - 1 ) / BCOPY_MEM_ALIGN ) * BCOPY_MEM_ALIGN )

template<typename block_t>
struct bcopy_ptr_arg_t {
  const Vector<block_t> & blocks;
  char* p_dst;
  const char* p_src;
};

template<typename T, typename blocks_t>
bool bcopy_host_host(size_t bs, size_t alignment, const std::vector<blocks_t> & arg) {
  
  if ((bs % sizeof(T) != 0) || (alignment % sizeof(T) != 0))
    return false;

  for (auto & a : arg) {
    if ((size_t)a.p_dst % sizeof(T) != 0 ||  // dst is not aligned w.r.t. T
	(size_t)a.p_src % sizeof(T) != 0)    // src is not aligned w.r.t. T
      return false;
  }

  ASSERT(BCOPY_MEM_ALIGN % sizeof(T) == 0); // make sure we update BCOPY_MEM_ALIGN if needed

  size_t npb = bs / sizeof(T);

  thread_region
    {
      for (auto & a : arg) {
	auto & b = a.blocks;
	
	T* dst = (T*)a.p_dst;
	const T* src = (const T*)a.p_src;
	
	size_t b_size = b.size();
	thread_for_in_region(i, npb * b_size, {
	    auto & x = b[i / npb];
	    size_t i_dst = x.start_dst / sizeof(T);
	    size_t i_src = x.start_src / sizeof(T);
	    size_t j = i % npb;
	
	    dst[i_dst + j] = src[i_src + j];
	  });
      }
  }
  return true;
}

accelerator_inline const TComplexF coalescedRead(const TComplexF & x) {
  return x;
}

accelerator_inline void coalescedWrite(TComplexF& x, const TComplexF & y) {
  x=y;
}

template<typename T, typename vT, typename blocks_t>
bool bcopy_accelerator_accelerator(size_t bs, size_t alignment, const std::vector<blocks_t> & arg) {

  if ((bs % sizeof(vT) != 0) || (alignment % sizeof(vT) != 0))
    return false;

  size_t npb = bs / sizeof(vT);

  for (auto & a : arg) {
    auto & b = a.blocks;
    auto * pb = &b[0];

    vT* dst = (vT*)a.p_dst;
    const vT* src = (const vT*)a.p_src;
    
    accelerator_for(i, npb * b.size(), sizeof(vT)/sizeof(T), {
	auto & x = pb[i / npb];
	size_t i_dst = x.start_dst / sizeof(vT);
	size_t i_src = x.start_src / sizeof(vT);
	size_t j = i % npb;
	
	coalescedWrite(dst[i_dst + j], coalescedRead(src[i_src + j]));
      });
  }
  return true;
}
