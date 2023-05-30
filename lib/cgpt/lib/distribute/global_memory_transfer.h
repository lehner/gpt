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
#include <unordered_map>

	
template<typename offset_t, typename rank_t, typename index_t>
global_memory_transfer<offset_t,rank_t,index_t>::global_memory_transfer(rank_t _rank, Grid_MPI_Comm _comm) :
  global_transfer<rank_t>(_rank, _comm), comm_buffers_type(mt_none) {
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::distribute_merge_into(thread_blocks_t & target, const thread_blocks_t & src) {
  target.insert(target.end(), src.begin(), src.end());
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::distribute_merge_into(blocks_t & target, const thread_blocks_t & src) {
  size_t target_size = target.size();
  size_t src_size = src.size();
  target.resize(src_size + target_size);
  thread_for(i, src_size, {
      target[i + target_size] = src[i];
    });
}

template<typename offset_t, typename rank_t, typename index_t>
template<typename K, typename V1, typename V2>
void global_memory_transfer<offset_t,rank_t,index_t>::distribute_merge_into(std::map<K,V1> & target, const std::map<K,V2> & src) {
  for (auto & x : src) {
    V1 & y = target[x.first];
    const V2 & z = x.second;
    distribute_merge_into(y, z);
  }
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::fill_blocks_from_view_pair(const view_t& dst,
										 const view_t& src,
										 bool local_only) {

  if (!dst.blocks.size() || !src.blocks.size()) {
    block_size = BCOPY_MEM_ALIGN * 1024; // a generous default
  } else {
    block_size = cgpt_gcd(dst.block_size, src.block_size);
  }

  if (!local_only) {
    block_size = this->global_gcd(block_size);
  }
  
  long dst_factor = dst.block_size / block_size;
  long src_factor = src.block_size / block_size;
  ASSERT(dst_factor * dst.blocks.size() == src_factor * src.blocks.size());

  ASSERT(dst.is_aligned());
  ASSERT(src.is_aligned());
  
  //cgpt_timer t("fill");

  //t("parallel_fill");

  size_t threads;
  thread_region
    {
      threads = thread_max();
    }
  typedef std::map< rank_pair_t, std::map< index_pair_t, thread_blocks_t > > thread_block_t;

  std::vector<thread_block_t> tblocks(threads);

  thread_region
    {
      size_t thread = thread_num();

      thread_for_in_region(i, dst_factor * dst.blocks.size(), {
	  size_t i_src = i / src_factor;
	  size_t i_dst = i / dst_factor;
	  
	  size_t j_src = i % src_factor;
	  size_t j_dst = i % dst_factor;
	  
	  auto & s = src.blocks[i_src];
	  auto & d = dst.blocks[i_dst];
	  
	  tblocks[thread]
	    [{d.rank, s.rank}]
	    [{d.index, s.index}]
	    .push_back({d.start + j_dst * block_size,s.start + j_src * block_size});
	});
    }

  //t("merge");
  
  // now merge neighboring thread data
  size_t merge_base = 2;
  while (merge_base < 2*threads) {
    thread_region
      {
	size_t thread = thread_num();
	if (thread % merge_base == 0) {
	  size_t partner = thread + merge_base / 2;

	  if (partner < threads)
	    distribute_merge_into(tblocks[thread], tblocks[partner]);
	}
      }
    merge_base *= 2;
  }

  //t("final");
  distribute_merge_into(blocks, tblocks[0]);

  //t.report();

  
}

template<typename offset_t, typename rank_t, typename index_t>
template<typename ranks_t>
void global_memory_transfer<offset_t,rank_t,index_t>::prepare_comm_message(comm_message & msg, ranks_t & ranks, bool populate) {

  if (!populate) {
    msg.reserve(sizeof(ranks.first) + sizeof(size_t));
    for (auto & indices : ranks.second)
      msg.reserve(sizeof(indices.first) + sizeof(size_t) + indices.second.size()*sizeof(block_t));
  } else {
    msg.put(ranks.first);

    size_t n = ranks.second.size();
    msg.put(n);
    
    for (auto & indices : ranks.second) {
      msg.put(indices.first);

      n = indices.second.size();
      msg.put(n);

      block_t* dst = (block_t*)msg.get(sizeof(block_t) * n);
      block_t* src = &indices.second[0];
      thread_for(i, n, {
	  dst[i] = src[i];
	});
    }
  }
  
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::gather_my_blocks() {

  // serialize my blocks to approprate communication buffers
  std::map<rank_t, comm_message> tasks_for_rank;

  Timer("gather: reserve comms");

  // reserve comm buffers
  for (auto & ranks : blocks) {
    if (ranks.first.dst_rank != this->rank)
      prepare_comm_message(tasks_for_rank[ranks.first.dst_rank], ranks, false);
    if ((ranks.first.dst_rank != ranks.first.src_rank) &&
	(ranks.first.src_rank != this->rank))
      prepare_comm_message(tasks_for_rank[ranks.first.src_rank], ranks, false);
  }

  Timer("gather: alloc comms");
  
  for (auto & x : tasks_for_rank) {
    x.second.alloc();
  }

  Timer("gather: prepare comm");
  for (auto & ranks : blocks) {
    if (ranks.first.dst_rank != this->rank)
      prepare_comm_message(tasks_for_rank[ranks.first.dst_rank], ranks, true);
    if ((ranks.first.dst_rank != ranks.first.src_rank) &&
	(ranks.first.src_rank != this->rank))
      prepare_comm_message(tasks_for_rank[ranks.first.src_rank], ranks, true);
  }

  Timer("gather: send_recv");

  std::map<rank_t, comm_message> tasks_from_rank;
  this->multi_send_recv(tasks_for_rank, tasks_from_rank);

  Timer("gather: merge");
  // de-serialize my blocks from appropriate communication buffers
  merge_comm_blocks(tasks_from_rank);

  Timer("gather: clean");
  
  // and finally remove all blocks from this rank in which it does not participate
  auto i = std::begin(blocks);
  while (i != std::end(blocks)) {
    if (i->first.dst_rank != this->rank &&
	i->first.src_rank != this->rank)
      i = blocks.erase(i);
    else
      ++i;
  }

}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::merge_comm_blocks(std::map<rank_t, comm_message> & all_src) {

  std::map< rank_pair_t, std::map< index_pair_t, abstract_blocks_t > > abs;
 
  for (auto & _src : all_src) {
    auto & src = _src.second;
    while (!src.eom()) {
      rank_t* rank = (rank_t*)src.get(sizeof(rank_t)*2);
      
      auto & r = abs[{rank[0],rank[1]}];
      auto & r0 = blocks[{rank[0],rank[1]}];
      
      size_t n_indices = *(size_t*)src.get(sizeof(size_t));
      for (size_t i=0;i<n_indices;i++) {
	index_t* index = (index_t*)src.get(sizeof(index_t)*2);
	
	auto & idx = r[{index[0],index[1]}];
	auto & idx0 = r0[{index[0],index[1]}];
	
	size_t n_blocks = *(size_t*)src.get(sizeof(size_t));

	if (idx.n_blocks == 0) {
	  idx.n_blocks = idx0.size();
	  idx.offset = 0;
	}
	
	idx.n_blocks += n_blocks;
	
	//block_t* src_block = (block_t*)
	src.get(sizeof(block_t) * n_blocks); // skip actual blocks for now
	
      }
    }
  }

  // first adjust existing blocks
  for (auto & x : abs) {
    auto & r = blocks[x.first];
    for (auto & y : x.second) {
      auto & idx0 = r[y.first];

      Vector<block_t> tmp(y.second.n_blocks);
      thread_for(i, idx0.size(), {
	  tmp[i] = idx0[i];
	});

      y.second.offset = idx0.size();
      idx0 = std::move(tmp);
    }
  }

  // now merge in new blocks
  for (auto & _src : all_src) {
    auto & src = _src.second;
    src.reset();
    while (!src.eom()) {
      rank_t* rank = (rank_t*)src.get(sizeof(rank_t)*2);
      
      auto & r = abs[{rank[0],rank[1]}];
      auto & r0 = blocks[{rank[0],rank[1]}];
      
      size_t n_indices = *(size_t*)src.get(sizeof(size_t));
      for (size_t i=0;i<n_indices;i++) {
	index_t* index = (index_t*)src.get(sizeof(index_t)*2);
	
	auto & idx = r[{index[0],index[1]}];
	auto & idx0 = r0[{index[0],index[1]}];
	
	size_t n_blocks = *(size_t*)src.get(sizeof(size_t));

	block_t* src_block = (block_t*)src.get(sizeof(block_t) * n_blocks);
	
	thread_for(i, n_blocks, {
	    idx0[i + idx.offset] = src_block[i];
	  });
	
	idx.offset += n_blocks;	
      }
    }
  }
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::optimize() {

  long global_gcd = -1;
  
  for (auto & ranks : blocks) {
    for (auto & indices : ranks.second) {
      long gcd = optimize(indices.second);
      if (global_gcd == -1)
	global_gcd = gcd;
      else
	global_gcd = cgpt_gcd(global_gcd, gcd);
    }
  }

  for (auto & a : recv_blocks) {
    for (auto & b : a.second) {
      long gcd = optimize(b.second);
      if (global_gcd == -1)
	global_gcd = gcd;
      else
	global_gcd = cgpt_gcd(global_gcd, gcd);
    }
  }
    
  for (auto & a : send_blocks) {
    for (auto & b : a.second) {
      long gcd = optimize(b.second);
      if (global_gcd == -1)
	global_gcd = gcd;
      else
	global_gcd = cgpt_gcd(global_gcd, gcd);
    }
  }

  // now skip
  if (global_gcd != 1) {
    for (auto & ranks : blocks)
      for (auto & indices : ranks.second)
	skip(indices.second, global_gcd);
    
    for (auto & a : recv_blocks)
      for (auto & b : a.second)
	skip(b.second, global_gcd);
    
    for (auto & a : send_blocks)
      for (auto & b : a.second)
	skip(b.second, global_gcd);
    
    block_size *= global_gcd;
  }
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::skip(blocks_t& blocks, long gcd) {

  ASSERT(blocks.size() % gcd == 0);
  size_t n = blocks.size() / gcd;

  Vector<block_t> tmp(n);
  thread_for(i, n, {
      tmp[i] = blocks[i*gcd];
    });

  blocks = std::move(tmp);
}

template<typename offset_t, typename rank_t, typename index_t>
long global_memory_transfer<offset_t,rank_t,index_t>::optimize(blocks_t& blocks) {

  struct {
    bool operator()(const block_t& a, const block_t& b) const
    {
      return a.start_dst < b.start_dst; // sort by destination address (better for first write page mapping)
      // Make this an option?
      //return a.start_src < b.start_src; // sort by source address (better for parallel transport)
    }
  } less;

  //cgpt_timer t("optimize");

  //t("sort"); // can make this 2x faster by first only extracting the sorting index
  cgpt_sort(blocks, less);

  //t("unique");
  Vector<block_t> unique_blocks;
  cgpt_sorted_unique(unique_blocks, blocks, [](const block_t & a, const block_t & b) {
					      return (a.start_dst == b.start_dst && a.start_src == b.start_src);
					    });

  //t("print");
  //std::cout << GridLogMessage << "Unique " << blocks.second.size() << " -> " << unique_blocks.size() << std::endl;
  
  //t("rle");
  Vector<size_t> start, repeats;
  size_t bs = block_size;
  long gcd = cgpt_rle(start, repeats, unique_blocks, [bs](const block_t & a, const block_t & b) {
						       return (a.start_dst + bs == b.start_dst &&
							       a.start_src + bs == b.start_src);
						     });


  //t("move");
  blocks = std::move(unique_blocks);

  //t.report();
  return gcd;

}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::create_bounds_and_alignment() {

  for (auto & ranks : blocks) {
    rank_t dst_rank = ranks.first.dst_rank;
    rank_t src_rank = ranks.first.src_rank;

    for (auto & indices : ranks.second) {
      index_t dst_idx = indices.first.dst_index;
      index_t src_idx = indices.first.src_index;
      if (dst_rank == this->rank && dst_idx >= bounds_dst.size())
	bounds_dst.resize(dst_idx+1,0);
      if (src_rank == this->rank && src_idx >= bounds_src.size())
	bounds_src.resize(src_idx+1,0);
      if (src_idx >= alignment.size())
	alignment.resize(src_idx+1,0);
      if (dst_idx >= alignment.size())
	alignment.resize(dst_idx+1,0);

      auto & a = indices.second;
      thread_region
	{
	  offset_t t_max_dst = 0;
	  offset_t t_max_src = 0;

	  offset_t alignment_src = alignment[src_idx];
	  offset_t alignment_dst = alignment[dst_idx];
	  
	  thread_for_in_region(i, a.size(), {
	      offset_t start_dst = a[i].start_dst;
	      offset_t start_src = a[i].start_src;
	      offset_t end_dst = start_dst + block_size;
	      offset_t end_src = start_src + block_size;
	      if (dst_rank == this->rank && end_dst > t_max_dst)
		t_max_dst = end_dst;
	      if (src_rank == this->rank && end_src > t_max_src)
		t_max_src = end_src;

	      alignment_src = (!alignment_src) ? start_src : cgpt_gcd(alignment_src, start_src);
	      alignment_dst = (!alignment_dst) ? start_dst : cgpt_gcd(alignment_dst, start_dst);
	    });

	  thread_critical
	    {
	      if (dst_rank == this->rank && t_max_dst > bounds_dst[dst_idx])
		bounds_dst[dst_idx] = t_max_dst;
	      if (src_rank == this->rank && t_max_src > bounds_src[src_idx])
		bounds_src[src_idx] = t_max_src;

	      alignment[src_idx] = (!alignment[src_idx]) ? alignment_src : cgpt_gcd(alignment_src, alignment[src_idx]);
	      alignment[dst_idx] = (!alignment[dst_idx]) ? alignment_dst : cgpt_gcd(alignment_dst, alignment[dst_idx]);
	    }
	}
    }
  }

  global_alignment = alignment.size() ? alignment[0] : 1;
  for (size_t i=1;i<alignment.size();i++)
    global_alignment = cgpt_gcd(global_alignment, alignment[i]);

}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::create(const view_t& _dst,
							     const view_t& _src,
							     memory_type use_comm_buffers_of_type,
							     bool local_only,
							     bool skip_optimize) {

  //{
  //  uint64_t d = 0;
  //  this->global_sum(&d,1);
  //}

  Timer("fill blocks");
  // reset
  blocks.clear();

  // fill
  fill_blocks_from_view_pair(_dst,_src,local_only);

  if (!local_only) {
    //t("sync");
    //{
    //  uint64_t d = 0;
    //  this->global_sum(&d,1);
    //}

    Timer("gather");
    // gather my blocks
    gather_my_blocks();
  }

  Timer("optimize");
  // optimize blocks after gathering all of my blocks
  if (!skip_optimize)
    optimize();

  if (!local_only) {

    Timer("create_com_buffers");
    // optionally create communication buffers
    create_comm_buffers(use_comm_buffers_of_type);
  }

  Timer("create_bounds_and_alignment");
  // create bounds and alignment information
  create_bounds_and_alignment();

 
  Timer();
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::create_comm_buffers(memory_type mt) {

  // first remove existing buffers
  send_buffers.clear();
  recv_buffers.clear();
  send_blocks.clear();
  recv_blocks.clear();
  comm_buffers_type = mt;

  // then exit if nothing to create
  if (mt == mt_none)
    return;

  // first get size of buffers
  std::map<rank_t, size_t> send_size, recv_size;
  std::map<rank_t, std::map< index_t, abstract_blocks_t > > send_abs_blocks, recv_abs_blocks;
  for (auto & ranks : blocks) {
    rank_t dst_rank = ranks.first.dst_rank;
    rank_t src_rank = ranks.first.src_rank;

    if (dst_rank == src_rank) {
      ASSERT(src_rank == this->rank);
      // no communication
      continue;
    }

    if (dst_rank == this->rank) {
      for (auto & indices : ranks.second) {
	auto& rb = recv_abs_blocks[src_rank][indices.first.dst_index]; // dst index
	rb.n_blocks += indices.second.size();
      }
    } else if (src_rank == this->rank) {
      for (auto & indices : ranks.second) {
	auto& sb = send_abs_blocks[dst_rank][indices.first.src_index]; // src index
	sb.n_blocks += indices.second.size();
      }
    } else {
      ERR("Mismatched comm info at rank %ld",this->rank);
    }
  }

  // now create copy blocks
  for (auto & ranks : blocks) {
    rank_t dst_rank = ranks.first.dst_rank;
    rank_t src_rank = ranks.first.src_rank;

    if (dst_rank == src_rank)
      continue;

    if (dst_rank == this->rank) {
      size_t sz = 0;
      for (auto & indices : ranks.second) {
	auto& rb0 = recv_abs_blocks[src_rank][indices.first.dst_index];
	auto& rb = recv_blocks[src_rank][indices.first.dst_index];

	if (rb0.offset == 0)
	  rb.resize(rb0.n_blocks);

	thread_for(i, indices.second.size(), {
	    rb[rb0.offset + i] = { indices.second[i].start_dst, sz + i * block_size };
	  });

	rb0.offset += indices.second.size();
	
	sz += BCOPY_ALIGN(block_size * indices.second.size());
      }
      recv_size[src_rank] += sz;
    } else if (src_rank == this->rank) {
      size_t sz = 0;
      for (auto & indices : ranks.second) {
	auto& sb0 = send_abs_blocks[dst_rank][indices.first.src_index];
	auto& sb = send_blocks[dst_rank][indices.first.src_index];

	if (sb0.offset == 0)
	  sb.resize(sb0.n_blocks);
	
	thread_for(i, indices.second.size(), {
	    sb[sb0.offset + i] = { sz + i * block_size, indices.second[i].start_src };
	  });

	sb0.offset += indices.second.size();
	
	sz += BCOPY_ALIGN(block_size * indices.second.size());	
      }
      send_size[dst_rank] += sz;
    }
  }

  // allocate buffers
  for (auto & s : send_size) {
    //printf("Rank %d has a send_buffer of size %d for rank %d\n",
    //	   this->rank, (int)s.second, (int)s.first);
    send_buffers.insert(std::make_pair(s.first,memory_buffer(s.second, mt)));
  }

  for (auto & s : recv_size) {
    //printf("Rank %d has a recv_buffer of size %d for rank %d\n",
    //	   this->rank, (int)s.second, (int)s.first);
    recv_buffers.insert(std::make_pair(s.first,memory_buffer(s.second, mt)));
  }
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::print() {

  for (auto & ranks : blocks) {
    for (auto & indices : ranks.second) {
      std::cout << GridLogMessage 
		<< "[" << ranks.first.dst_rank << "," << ranks.first.src_rank << "]"
		<< "[" << indices.first.src_index << "," << indices.first.src_index << "] : " << std::endl;

      for (auto & block : indices.second) {
	std::cout << GridLogMessage << block.start_dst << " <- " <<
	  block.start_src << " for " << block_size << std::endl;
      }
    }
  }
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::bcopy(const std::vector<bcopy_arg_t>& args) {

  std::unordered_map<size_t,std::vector<bcopy_ptr_arg_t<block_t>>> bca;

#define bcopy_map_idx(mt_dst, mt_src) (mt_dst * mt_int_len + mt_src)

  for (auto & arg : args) {
    auto & base_dst = arg.base_dst;
    auto & base_src = arg.base_src;
    auto & blocks   = arg.blocks;

    memory_type mt_dst = base_dst.type;
    char* p_dst = (char*)base_dst.ptr;
    
    memory_type mt_src = base_src.type;
    const char* p_src = (const char*)base_src.ptr;

    bca[bcopy_map_idx(mt_dst,mt_src)].push_back({blocks, p_dst, p_src});
  }

  for (auto & bcc : bca) {
    size_t idx = bcc.first;
    auto & bc = bcc.second;

    switch (idx) {
    case mt_host * mt_int_len + mt_host:
      if (bcopy_host_host<vComplexF>(block_size,global_alignment,bc));
      else if (bcopy_host_host<ComplexF>(block_size,global_alignment,bc));
      else if (bcopy_host_host<double>(block_size,global_alignment,bc));
      else if (bcopy_host_host<float>(block_size,global_alignment,bc));
      else {
	ERR("No fast copy method for block size %ld host<>host implemented", (long)block_size);
      }
      break;
    case mt_host * mt_int_len + mt_accelerator:
      for (auto & bi : bc) {
	for (size_t i=0;i<bi.blocks.size();i++) {
	  auto&b=bi.blocks[i];
	  acceleratorCopyFromDevice((void*)&bi.p_src[b.start_src],(void*)&bi.p_dst[b.start_dst],block_size);
	}
      }
      break;
    case mt_accelerator * mt_int_len + mt_host:
      for (auto & bi : bc) {
	for (size_t i=0;i<bi.blocks.size();i++) {
	  auto&b=bi.blocks[i];
	  acceleratorCopyToDevice((void*)&bi.p_src[b.start_src],(void*)&bi.p_dst[b.start_dst],block_size);
	}
      }
      break;
    case mt_accelerator * mt_int_len + mt_accelerator:
      if (bcopy_accelerator_accelerator<SpinMatrixF,vSpinMatrixF>(block_size,global_alignment,bc));
      else if (bcopy_accelerator_accelerator<ColourMatrixF,vColourMatrixF>(block_size,global_alignment,bc));
      else if (bcopy_accelerator_accelerator<SpinVectorF,vSpinVectorF>(block_size,global_alignment,bc));
      else if (bcopy_accelerator_accelerator<ColourVectorF,vColourVectorF>(block_size,global_alignment,bc));
      else if (bcopy_accelerator_accelerator<TComplexF,vTComplexF>(block_size,global_alignment,bc));
      else if (bcopy_accelerator_accelerator<TComplexF,TComplexF>(block_size,global_alignment,bc)); // fallback option
      else {
	ERR("No fast copy method for block size %ld accelerator<>accelerator implemented", (long)block_size);
      }
      break;
    }
  }

}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::execute(std::vector<memory_view>& base_dst, 
							      std::vector<memory_view>& base_src) {

  // first check bounds
  ASSERT(base_dst.size() >= bounds_dst.size());
  ASSERT(base_src.size() >= bounds_src.size());

  for (size_t i=0;i<bounds_dst.size();i++) {
    if (base_dst[i].sz < bounds_dst[i]) {
      ERR("Destination view index %ld is too small (%ld < %ld)", i, base_dst[i].sz, bounds_dst[i]);
    }
  }

  for (size_t i=0;i<bounds_src.size();i++) {
    if (base_src[i].sz < bounds_src[i]) {
      ERR("Source view index %ld is too small (%ld < %ld)", i, base_src[i].sz, bounds_src[i]);
    }
  }

  //{
  //  uint64_t d = 0;
  //  this->global_sum(&d,1);
  //}
  //cgpt_timer tt("execute");
  //tt("pre");
  
  // if there is no buffer, directly issue separate isend / irecv for each block
  int stats_isends = 0;
  int stats_irecvs = 0;
  size_t stats_send_bytes = 0;
  size_t stats_recv_bytes = 0;
  
  if (comm_buffers_type == mt_none) {

    // first start remote submissions
    for (auto & ranks : blocks) {
      rank_t dst_rank = ranks.first.dst_rank;
      rank_t src_rank = ranks.first.src_rank;

      if (src_rank == this->rank && dst_rank != this->rank) {
	for (auto & indices : ranks.second) {
	  index_t src_idx = indices.first.src_index;

	  for (auto & block : indices.second) {
	    this->isend(dst_rank, (char*)base_src[src_idx].ptr + block.start_src, block_size);
	  }
	}
      } else if (src_rank != this->rank && dst_rank == this->rank) {
	for (auto & indices : ranks.second) {
	  index_t dst_idx = indices.first.dst_index;

	  for (auto & block : indices.second) {
	    this->irecv(src_rank, (char*)base_dst[dst_idx].ptr + block.start_dst, block_size);
	  }
	}
      }
    }

  } else {

    std::vector<bcopy_arg_t> bca;
    
    // if there is a buffer, first gather in communication buffer
    for (auto & ranks : send_blocks) {
      rank_t dst_rank = ranks.first;
      auto & dst = send_buffers.at(dst_rank).view;

      for (auto & indices : ranks.second) {
	index_t src_idx = indices.first;
	bca.push_back({indices.second, dst, base_src[src_idx]});
      }
    }

    bcopy(bca);

    // send/recv buffers
    for (auto & buf : send_buffers) {
      this->isend(buf.first, buf.second.view.ptr, buf.second.view.sz);
      stats_isends += 1;
      stats_send_bytes += buf.second.view.sz;
    }
    for (auto & buf : recv_buffers) {
      this->irecv(buf.first, buf.second.view.ptr, buf.second.view.sz);
      stats_irecvs += 1;
      stats_recv_bytes += buf.second.view.sz;
    }
  }


  // then do local copies
  //tt("local");
  {
    std::vector<bcopy_arg_t> bca;
    for (auto & ranks : blocks) {
      rank_t dst_rank = ranks.first.dst_rank;
      rank_t src_rank = ranks.first.src_rank;
      
      if (src_rank == this->rank && dst_rank == this->rank) {
	for (auto & indices : ranks.second) {
	  index_t dst_idx = indices.first.dst_index;
	  index_t src_idx = indices.first.src_index;
	  
	  bca.push_back({indices.second,base_dst[dst_idx],base_src[src_idx]});
	}
      }
    }
    bcopy(bca);
  }
  
  //tt("wait");
  // then wait for remote copies to finish
  //double t0 = cgpt_time();
  this->waitall();
  //double t1 = cgpt_time();
  //std::cout << GridLogMessage << "WAIT: time " << t1-t0 << " sends " << stats_isends <<
  //  " bytes " << stats_send_bytes << " recvs " << stats_irecvs << " bytes " << stats_recv_bytes << std::endl;
  

  //tt("post");
  // if buffer was used, need to re-distribute locally
  if (comm_buffers_type != mt_none) {
    std::vector<bcopy_arg_t> bca;
    for (auto & ranks : recv_blocks) {

      rank_t src_rank = ranks.first;
      auto & src = recv_buffers.at(src_rank).view;

      for (auto & indices : ranks.second) {
	index_t dst_idx = indices.first;
	//printf("On rank %d recv_block from %d block %d\n",
	//     (int)this->rank,
	//     (int)src_rank,
	//     (int)buf.sz);
	//for (size_t i=0;i<buf.sz/sizeof(double);i++)
	//  printf("%d = %g\n",(int)i,((double*)buf.ptr)[i]);
	bca.push_back({indices.second, base_dst[dst_idx], src});
      }
    }
    bcopy(bca);
  }

  //tt.report();

}
