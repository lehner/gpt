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
template<typename offset_t, typename rank_t, typename index_t>
void global_memory_view<offset_t,rank_t,index_t>::print() const {
  std::cout << "global_memory_view:" << std::endl;
  for (size_t i=0;i<blocks.size();i++) {
    auto & bc = blocks[i];
    std::cout << " [" << i << "/" << blocks.size() << "] = { " << bc.rank << ", " << bc.index << ", " << bc.start << ", " << block_size << " }" << std::endl;
  }
}

template<typename offset_t, typename rank_t, typename index_t>
offset_t global_memory_view<offset_t,rank_t,index_t>::size() const {
  return block_size * blocks.size();
}

template<typename offset_t, typename rank_t, typename index_t>
bool global_memory_view<offset_t,rank_t,index_t>::is_aligned() const {
  bool aligned = true;
  thread_region
    {
      bool thread_aligned = true;
      thread_for_in_region(i, blocks.size(), {
	  if (blocks[i].start % block_size)
	    thread_aligned = false;
	});
      thread_critical
	{
	  if (!thread_aligned)
	    aligned = false;
	}
    }
  return aligned;
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_view<offset_t,rank_t,index_t>::operator=(const global_memory_view<offset_t,rank_t,index_t>& other) {
  block_size = other.block_size;
  blocks.resize(other.blocks.size());
  thread_for(i, blocks.size(), {
      blocks[i] = other.blocks[i];
    });
}
