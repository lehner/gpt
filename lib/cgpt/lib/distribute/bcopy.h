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

template<typename T, typename blocks_t>
bool bcopy_host_host(const blocks_t& blocks, char* p_dst, const char* p_src) {
  size_t bs = blocks.first;
  if (bs % sizeof(T) != 0 ||             // block is not multiple of T
      (size_t)p_dst % sizeof(T) != 0 ||  // dst is not aligned w.r.t. T
      (size_t)p_src % sizeof(T) != 0)    // src is not aligned w.r.t. T
    return false;

  ASSERT(BCOPY_MEM_ALIGN % sizeof(T) == 0); // make sure we update BCOPY_MEM_ALIGN if needed

  size_t npb = bs / sizeof(T);

  auto & b = blocks.second;

  T* dst = (T*)p_dst;
  const T* src = (const T*)p_src;

  size_t b_size = b.size();
  thread_for(i, npb * b_size, {
      auto & x = b[i / npb];
      size_t i_dst = x.start_dst / sizeof(T);
      size_t i_src = x.start_src / sizeof(T);
      size_t j = i % npb;
      
      dst[i_dst + j] = src[i_src + j];
    });

  return true;
}

template<typename T, typename blocks_t>
bool bcopy_accelerator_accelerator(const blocks_t& blocks, char* p_dst, const char* p_src) {
  size_t bs = blocks.first;
  if (bs % sizeof(T) != 0)
    return false;

  size_t npb = bs / sizeof(T);

  auto & b = blocks.second;

  T* dst = (T*)p_dst;
  const T* src = (const T*)p_src;

  accelerator_for(i, npb * b.size(), {
      auto & x = b[i / npb];
      size_t i_dst = x.start_dst / sizeof(T);
      size_t i_src = x.start_src / sizeof(T);
      size_t j = i % npb;
      
      dst[i_dst + j] = src[i_src + j];
    });
  
  return true;
}
