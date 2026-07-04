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
#include "../copy.h"

class cgpt_copy_job : public cgpt_kernel_job_base {
 public:

  gm_transfer* plan;
  std::vector<gm_transfer::memory_view> vdst, vsrc;
  
  cgpt_copy_job(gm_transfer* _plan,
		const std::vector<gm_transfer::memory_view>& _vdst,
		const std::vector<gm_transfer::memory_view>& _vsrc) : plan(_plan),
								      vdst(_vdst),
								      vsrc(_vsrc)
  {
  }

  std::string description() {
    std::ostringstream oss;

    oss << "COPY(block_size=" << plan->block_size << ", align=" << plan->global_alignment;

    for (auto & rank : plan->blocks) {
      auto rank_dst = rank.first.dst_rank;
      auto rank_src = rank.first.src_rank;

      size_t total_size = 0;
      for (auto & index : rank.second) {
	auto index_dst = index.first.dst_index;
	auto index_src = index.first.src_index;
	size_t blocks = index.second.size();
	size_t size = plan->block_size * blocks;
	total_size += size;
      }

      oss << " | " << total_size << " bytes " << rank_src << " -> " << rank_dst;
    }

    oss << ")";
    return oss.str();
  }
  
  virtual ~cgpt_copy_job() {
  }

  virtual void execute(GridBLAS& blas) {
    blas.synchronise();
    plan->execute(vdst, vsrc);
  }
};
