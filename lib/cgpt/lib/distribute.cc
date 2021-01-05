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
#include <unordered_map>

#define BCOPY_MEM_ALIGN   (sizeof(vComplexF))

//
// global_transfer
//
template<typename rank_t>
global_transfer<rank_t>::global_transfer(rank_t _rank, Grid_MPI_Comm _comm) : rank(_rank), comm(_comm) {

#ifdef CGPT_USE_MPI
  MPI_Comm_size(comm,&mpi_ranks);
  MPI_Comm_rank(comm,&mpi_rank);
  mpi_rank_map.resize(mpi_ranks,0);
  ASSERT(rank < mpi_ranks);
  mpi_rank_map[rank] = mpi_rank;
  ASSERT(MPI_SUCCESS == MPI_Allreduce(MPI_IN_PLACE,&mpi_rank_map[0],mpi_ranks * sizeof(rank_t) / sizeof(int),MPI_INT,MPI_SUM,comm));
#else
  mpi_ranks=1;
  mpi_rank=0;
  mpi_rank_map.push_back(0);
  ASSERT(rank == 0);
#endif
}

template<typename rank_t>
void global_transfer<rank_t>::global_sum(std::vector<uint64_t>& data) {
#ifdef CGPT_USE_MPI
  ASSERT(MPI_SUCCESS == MPI_Allreduce(MPI_IN_PLACE,&data[0],data.size(),MPI_UINT64_T,MPI_SUM,comm));
#endif
}

template<typename rank_t>
template<typename data_t>
void global_transfer<rank_t>::root_to_all(const std::map<rank_t, std::vector<data_t> > & all, std::vector<data_t>& my) {

  // store mine
  if (rank == 0) {
    auto e = all.find(0);
    my.clear();
    if (e != all.end())
      my = e->second;
  }

#ifdef CGPT_USE_MPI

  std::vector<int> all_size(mpi_ranks,0);
  int my_size;
  for (rank_t i=0; i<mpi_ranks; i++) {
    auto e = all.find(i);
    all_size[i] = (e != all.end()) ? (int)e->second.size() : 0;
  }
  ASSERT(MPI_SUCCESS == MPI_Scatter(&all_size[0], 1, MPI_INT, &my_size, 1, MPI_INT, mpi_rank_map[0], comm));

  // root node now receives from every node the list of its partners (if it is non-vanishing)
  std::vector<MPI_Request> req;
  if (rank == 0) {
    // root node now
    for (rank_t i=1;i<mpi_ranks;i++) {

      int rank_size = all_size[mpi_rank_map[i]];

      if (rank_size != 0) {
        auto & data = all.at(i);

	isend(i,data);
      }

    }

  } else {

    if (my_size != 0) {
      my.resize(my_size);
      irecv(0,my);
    }
  }

  waitall();
#endif
}

template<typename rank_t>
template<typename data_t>
void global_transfer<rank_t>::all_to_root(const std::vector<data_t>& my, std::map<rank_t, std::vector<data_t> > & all) {

  // store mine
  if (rank == 0)
    all[0] = my;

#ifdef CGPT_USE_MPI
  int my_size = (int)my.size();
  std::vector<int> all_size(mpi_ranks,0);
  ASSERT(MPI_SUCCESS == MPI_Gather(&my_size, 1, MPI_INT, &all_size[0], 1, MPI_INT, mpi_rank_map[0], comm));

  // root node now receives from every node the list of its partners (if it is non-vanishing)
  if (rank == 0) {
    // root node now
    for (rank_t i=1;i<mpi_ranks;i++) {

      int rank_size = all_size[mpi_rank_map[i]];

      if (rank_size != 0) {
	std::vector<data_t> & data = all[i];
	data.resize(rank_size);

	irecv(i,data);
      }

    }

  } else {

    if (my_size != 0) {
      isend(0,my);
    }
  }

  waitall();
#endif
}

template<typename rank_t>
void global_transfer<rank_t>::waitall() {
#ifdef CGPT_USE_MPI
  //printf("WAIT %d\n",(int)requests.size());
  if (!requests.size())
    return;
  std::vector<MPI_Status> stat(requests.size());
  ASSERT(MPI_SUCCESS == MPI_Waitall((int)requests.size(), &requests[0], &stat[0]));
  requests.clear();
#endif
}

template<typename rank_t>
void global_transfer<rank_t>::isend(rank_t other_rank, const void* pdata, size_t sz) {
  if (sz <= size_mpi_max) {
#ifdef CGPT_USE_MPI
    //printf("Send from %d to %d, %d bytes from %p (%g double)\n",this->rank,other_rank,(int)sz,pdata,*(double*)pdata);
    MPI_Request r;
    ASSERT(MPI_SUCCESS == MPI_Isend(pdata,sz,MPI_CHAR,mpi_rank_map[other_rank],0x3,comm,&r));
    requests.push_back(r);
#endif
  } else {
    while (sz) {
      size_t sz_block = std::min(sz,size_mpi_max);
      isend(other_rank,pdata,sz_block);
      sz -= sz_block;
      pdata = (void*)((char*)pdata + sz_block);
    }
  }
}

template<typename rank_t>
void global_transfer<rank_t>::irecv(rank_t other_rank, void* pdata, size_t sz) {
  if (sz <= size_mpi_max) {
#ifdef CGPT_USE_MPI
    //printf("Recv from %d to %d, %d bytes to %p\n",other_rank,this->rank,(int)sz,pdata);
    MPI_Request r;
    ASSERT(MPI_SUCCESS == MPI_Irecv(pdata,sz,MPI_CHAR,mpi_rank_map[other_rank],0x3,comm,&r));
    requests.push_back(r);
#endif
  } else {
    while (sz) {
      size_t sz_block = std::min(sz,size_mpi_max);
      irecv(other_rank,pdata,sz_block);
      sz -= sz_block;
      pdata = (void*)((char*)pdata + sz_block);
    }
  }
}

template<typename rank_t>
void global_transfer<rank_t>::provide_my_receivers_get_my_senders(const std::map<rank_t, size_t>& receivers,
								  std::map<rank_t, size_t>& senders) {

  struct rank_size_t {
    rank_t rank;
    size_t size;
  };

  // root node collects
  std::map<rank_t, std::vector<rank_size_t> > ranks_that_will_receive_data_from_rank;
  std::vector<rank_size_t> ranks_that_will_receive_my_data;
  for (auto & r : receivers) {
    ranks_that_will_receive_my_data.push_back({r.first, r.second});
  }

  all_to_root(ranks_that_will_receive_my_data, ranks_that_will_receive_data_from_rank);

  // create communication matrix
  std::map<rank_t, std::vector<rank_size_t> > ranks_from_which_rank_will_receive_data;
  if (this->rank == 0) {
    // for each rank create list of ranks that needs to talk to them
    std::map<rank_t, std::map<rank_t,size_t> > ranks_set_from_which_rank_will_receive_data;

    rank_t l;
    for (l=0;l<this->mpi_ranks;l++) {
      for (rank_size_t j : ranks_that_will_receive_data_from_rank[l]) {
	auto & x = ranks_set_from_which_rank_will_receive_data[j.rank];
	auto y = x.find(l);
	ASSERT(y == x.end());
	x[l] = j.size;
      }
    }

    for (l=0;l<this->mpi_ranks;l++) {
      for (auto & x : ranks_set_from_which_rank_will_receive_data[l]) {
	ranks_from_which_rank_will_receive_data[l].push_back({x.first,x.second});
      }
    }
  }

  // scatter packet number to be received by each rank
  std::vector<rank_size_t> ranks_from_which_I_will_receive_data;
  root_to_all(ranks_from_which_rank_will_receive_data, ranks_from_which_I_will_receive_data);

  // convert
  for (auto & r : ranks_from_which_I_will_receive_data) {
    //std::cout << "Rank " << this->rank << " here, will receive " << r.size << " bytes from rank " << r.rank << std::endl;
    senders[r.rank] = r.size;
  }
}

template<typename rank_t>
void global_transfer<rank_t>::multi_send_recv(const std::map<rank_t, comm_message>& send,
					      std::map<rank_t, comm_message>& recv) {

  // prepare list of all ranks for which I have data
  std::map<rank_t, size_t> my_receivers;
  std::map<rank_t, size_t> my_senders;

  for (auto & s : send) {
    ASSERT(s.first != rank); // should not send to myself
    my_receivers[s.first] = s.second.data.size();
  }

  provide_my_receivers_get_my_senders(my_receivers,my_senders);

  // allocate receive buffers
  for (auto & s : my_senders) {
    recv[s.first].data.resize(s.second);
  }

  // initiate communication
  for (auto & s : send) {
    isend(s.first, s.second.data);
  }

  for (auto & r : recv) {
    irecv(r.first, r.second.data);
  }

  // wait
  waitall();
}
//
// global_memory_view
//
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

template<typename V>
void cgpt_distribute_merge_into(std::vector<V> & target, const std::vector<V> & src) {
  target.insert(target.end(), src.begin(), src.end());
}

template<typename K, typename V>
void cgpt_distribute_merge_into(std::pair<K,V> & target, const std::pair<K,V> & src) {
  target.first = src.first;
  cgpt_distribute_merge_into(target.second, src.second);
}

template<typename K, typename V>
void cgpt_distribute_merge_into(std::map<K,V> & target, const std::map<K,V> & src) {
  for (auto & x : src) {
    V & y = target[x.first];
    const V & z = x.second;
    cgpt_distribute_merge_into(y, z);
  }
}
		
template<typename offset_t, typename rank_t, typename index_t>
global_memory_transfer<offset_t,rank_t,index_t>::global_memory_transfer(rank_t _rank, Grid_MPI_Comm _comm) :
  global_transfer<rank_t>(_rank, _comm), comm_buffers_type(mt_none) {
}


template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::fill_blocks_from_view_pair(const view_t& dst,
										 const view_t& src) {


  long block_size = cgpt_gcd(dst.block_size, src.block_size);
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
  typedef std::map< std::pair<rank_t,rank_t>, std::map< std::pair<index_t,index_t>, blocks_t > > thread_block_t;
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
	    [std::make_pair(d.rank, s.rank)]
	    [std::make_pair(d.index, s.index)]
	    .second.push_back({d.start + j_dst * block_size,s.start + j_src * block_size});
	});

      for (auto & b : tblocks[thread]) {
	for (auto & c : b.second) {
	  c.second.first = block_size;
	}
      }
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
	    cgpt_distribute_merge_into(tblocks[thread], tblocks[partner]);
	}
      }
    merge_base *= 2;
  }

  //t("final");
  blocks = tblocks[0];

  //t.report();

  
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::gather_my_blocks() {

  // serialize my blocks to approprate communication buffers
  std::map<rank_t, comm_message> tasks_for_rank;
  for (auto & ranks : blocks) {
    if (ranks.first.first != this->rank)
      tasks_for_rank[ranks.first.first].put(ranks);
    if ((ranks.first.first != ranks.first.second) &&
	(ranks.first.second != this->rank))
      tasks_for_rank[ranks.first.second].put(ranks);
  }

  std::map<rank_t, comm_message> tasks_from_rank;
  this->multi_send_recv(tasks_for_rank, tasks_from_rank);

  // de-serialize my blocks from appropriate communication buffers
  for (auto & tasks : tasks_from_rank) {
    while (!tasks.second.eom()) {
      std::pair< std::pair<rank_t,rank_t>, std::map< std::pair<index_t,index_t>, blocks_t > > ranks;
      tasks.second.get(ranks);

      // and merge with current blocks
      for (auto & idx : ranks.second) {
	merge_blocks(blocks[ranks.first][idx.first], idx.second);
      }
    }
  }

  // and finally remove all blocks from this rank in which it does not participate
  auto i = std::begin(blocks);
  while (i != std::end(blocks)) {
    if (i->first.first != this->rank &&
	i->first.second != this->rank)
      i = blocks.erase(i);
    else
      ++i;
  }

}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::merge_blocks(blocks_t& dst, const blocks_t& src) {

  if (!dst.second.size()) {
    dst = src;
    return;
  }

  offset_t block_size = (offset_t)cgpt_gcd(dst.first, src.first);

  //std::cout << GridLogMessage << "merge_blocks " << dst.first << " x " << src.first << " -> " << block_size << std::endl;
  offset_t dst_factor = dst.first / block_size;
  offset_t src_factor = src.first / block_size;
  
  blocks_t nd;
  nd.first = block_size;
  nd.second.resize(dst.second.size() * dst_factor + src.second.size() * src_factor);

  thread_for(i, dst.second.size(), {
      for (size_t j=0;j<dst_factor;j++) {
	auto & d = nd.second[i*dst_factor + j];
	auto & s = dst.second[i];
	d = s;
	d.start_dst += j*block_size;
	d.start_src += j*block_size;
      }
    });

  size_t o = dst.second.size() * dst_factor;
  
  thread_for(i, src.second.size(), {
      for (size_t j=0;j<src_factor;j++) {
	auto & d = nd.second[o + i*src_factor + j];
	auto & s = src.second[i];
	d = s;
	d.start_dst += j*block_size;
	d.start_src += j*block_size;
      }
    });
  
  dst.first = nd.first;
  dst.second = std::move(nd.second);
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::optimize() {
  for (auto & ranks : blocks) {
    for (auto & indices : ranks.second) {
      optimize(indices.second);
    }
  }
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::optimize(blocks_t& blocks) {

  struct {
    bool operator()(const block_t& a, const block_t& b) const
    {
      return a.start_dst < b.start_dst; // sort by destination address (better for first write page mapping)
    }
  } less;

  //cgpt_timer t("optimize");

  //t("sort"); // can make this 2x faster by first only extracting the sorting index
  cgpt_sort(blocks.second, less);

  //t("unique");
  std::vector<block_t> unique_blocks;
  cgpt_sorted_unique(unique_blocks, blocks.second, [](const block_t & a, const block_t & b) {
						     return (a.start_dst == b.start_dst && a.start_src == b.start_src);
						   });

  //t("print");
  //std::cout << GridLogMessage << "Unique " << blocks.second.size() << " -> " << unique_blocks.size() << std::endl;
  
  //t("rle");
  std::vector<size_t> start, repeats;
  size_t block_size = blocks.first;
  size_t gcd = cgpt_rle(start, repeats, unique_blocks, [block_size](const block_t & a, const block_t & b) {
						  return (a.start_dst + block_size == b.start_dst &&
							  a.start_src + block_size == b.start_src);
						});

  // can adjust block_size?
  if (gcd == 1) {
    //t("adopt unique_blocks");
    blocks.second = unique_blocks;
  } else {
    //t("adjust block_size");
    blocks.first *= gcd;
    ASSERT(unique_blocks.size() % gcd == 0); // this should always be true unless cgpt_rle made a mistake
    size_t n = unique_blocks.size() / gcd;
    blocks.second.resize(n);
    thread_for(i, n, {
	blocks.second[i] = unique_blocks[i*gcd];
      });
  }
  
  //t("print");
  //std::cout << GridLogMessage << "RLE " << start.size() << " gcd = " << gcd << std::endl;
  //std::cout << GridLogMessage << start[0] << " x " << repeats[0] <<  " last " << start[start.size()-1] << " x " << repeats[repeats.size()-1] << std::endl;
  //if (start.size() > 2) {
  //  std::cout << GridLogMessage << start[1] << " x " << repeats[1] << std::endl;
  //  std::cout << GridLogMessage << start[2] << " x " << repeats[2] << std::endl;
  //  for (size_t i=0;i<32;i++) {
  //	std::cout << GridLogMessage << i << ": " << unique_blocks[i].start_dst << "/" << unique_blocks[i].start_src << std::endl;
  //  }
  //}
  //t.report();

}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::create_bounds() {

  for (auto & ranks : blocks) {
    rank_t dst_rank = ranks.first.first;
    rank_t src_rank = ranks.first.second;

    for (auto & indices : ranks.second) {
      index_t dst_idx = indices.first.first;
      index_t src_idx = indices.first.second;
      if (dst_rank == this->rank && dst_idx >= bounds_dst.size())
	bounds_dst.resize(dst_idx+1,0);
      if (src_rank == this->rank && src_idx >= bounds_src.size())
	bounds_src.resize(src_idx+1,0);

      auto & a = indices.second.second;
      thread_region
	{
	  offset_t t_max_dst = 0;
	  offset_t t_max_src = 0;
	  
	  thread_for_in_region(i, a.size(), {
	      offset_t end_dst = a[i].start_dst + indices.second.first;
	      offset_t end_src = a[i].start_src + indices.second.first;
	      if (dst_rank == this->rank && end_dst > t_max_dst)
		t_max_dst = end_dst;
	      if (src_rank == this->rank && end_src > t_max_src)
		t_max_src = end_src;
	    });

	  thread_critical
	    {
	      if (dst_rank == this->rank && t_max_dst > bounds_dst[dst_idx])
		bounds_dst[dst_idx] = t_max_dst;
	      if (src_rank == this->rank && t_max_src > bounds_src[src_idx])
		bounds_src[src_idx] = t_max_src;
	    }
	}
    }
  }

}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::create(const view_t& _dst,
							     const view_t& _src,
							     memory_type use_comm_buffers_of_type,
							     bool local_only,
							     bool skip_optimize) {

  //cgpt_timer t("create");

  //t("fill blocks");
  // reset
  blocks.clear();

  // fill
  fill_blocks_from_view_pair(_dst,_src);

  //t("opt1");
  // optimize blocks obtained from this rank
  if (!skip_optimize)
    optimize();

  if (!local_only) {
    //t("gather");
    // gather my blocks
    gather_my_blocks();

    //t("opt2");
    // optimize blocks after gathering all of my blocks
    if (!skip_optimize)
      optimize();

    //t("createcom");
    // optionally create communication buffers
    create_comm_buffers(use_comm_buffers_of_type);
  }

  //t("createbounds");
  // create bounds
  create_bounds();

  //t.report();
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

  // Need to compute size of communications buffer for each target node
  // Also need to create contiguous send_blocks and recv_blocks
  std::map<rank_t, size_t> send_size, recv_size;
  for (auto & ranks : blocks) {
    rank_t dst_rank = ranks.first.first;
    rank_t src_rank = ranks.first.second;

    if (dst_rank == src_rank) {
      ASSERT(src_rank == this->rank);
      // no communication
      continue;
    }

    if (dst_rank == this->rank) {
      size_t sz = 0;
      for (auto & indices : ranks.second) {
	auto& rb = recv_blocks[src_rank][indices.first.first]; // dst index

	blocks_t nb;
	nb.first = indices.second.first;
	auto & a = indices.second.second;
	nb.second.resize(a.size());
	thread_for(i, nb.second.size(), {
	    nb.second[i] = { a[i].start_dst, sz + i * nb.first };
	  });
	sz += nb.first * nb.second.size();

	if (sz % BCOPY_MEM_ALIGN)
	  sz += BCOPY_MEM_ALIGN - sz % BCOPY_MEM_ALIGN;

	merge_blocks(rb, nb);
      }
      recv_size[src_rank] += sz;
    } else if (src_rank == this->rank) {
      size_t sz = 0;
      for (auto & indices : ranks.second) {
	auto& sb = send_blocks[dst_rank][indices.first.second]; // src index

	blocks_t nb;
	nb.first = indices.second.first;
	auto & a = indices.second.second;
	nb.second.resize(a.size());
	thread_for(i, nb.second.size(), {
	    auto & block = a[i];
	    nb.second[i] = { sz + i * nb.first, a[i].start_src };
	  });
	sz += nb.first * nb.second.size();

	if (sz % BCOPY_MEM_ALIGN)
	  sz += BCOPY_MEM_ALIGN - sz % BCOPY_MEM_ALIGN;

	merge_blocks(sb, nb);
      }
      send_size[dst_rank] += sz;
    } else {
      ERR("Mismatched comm info at rank %ld",this->rank);
    }
  }

  // optimize blocks
  for (auto & a : recv_blocks) {
    for (auto & b : a.second) {
      optimize(b.second);
    }
  }

  for (auto & a : recv_blocks) {
    for (auto & b : a.second) {
      optimize(b.second);
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
		<< "[" << ranks.first.first << "," << ranks.first.second << "]"
		<< "[" << indices.first.first << "," << indices.first.second << "] : " << std::endl;

      for (auto & block : indices.second.second) {
	std::cout << GridLogMessage << block.start_dst << " <- " <<
	  block.start_src << " for " << indices.second.first << std::endl;
      }
    }
  }
}

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

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::bcopy(const blocks_t& blocks,
							    memory_view& base_dst, 
							    const memory_view& base_src) {
  
  memory_type mt_dst = base_dst.type;
  char* p_dst = (char*)base_dst.ptr;

  memory_type mt_src = base_src.type;
  const char* p_src = (const char*)base_src.ptr;
  
  if (mt_dst == mt_host && mt_src == mt_host) {
    if (bcopy_host_host<vComplexF>(blocks, p_dst, p_src));
    else if (bcopy_host_host<ComplexF>(blocks, p_dst, p_src));
    else if (bcopy_host_host<double>(blocks, p_dst, p_src));
    else if (bcopy_host_host<float>(blocks, p_dst, p_src));
    else {
      ERR("No fast copy method for block size %ld host<>host implemented", (long)blocks.first);
    }
  } else if (mt_dst == mt_host && mt_src == mt_accelerator) {
    for (size_t i=0;i<blocks.second.size();i++) {
      auto&b=blocks.second[i];
      acceleratorCopyFromDevice((void*)&p_src[b.start_src],(void*)&p_dst[b.start_dst],blocks.first);
    }
  } else if (mt_dst == mt_accelerator && mt_src == mt_host) {
    for (size_t i=0;i<blocks.second.size();i++) {
      auto&b=blocks.second[i];
      acceleratorCopyToDevice((void*)&p_src[b.start_src],(void*)&p_dst[b.start_dst],blocks.first);
    }
  } else if (mt_dst == mt_accelerator && mt_src == mt_accelerator) {
    if (bcopy_accelerator_accelerator<vComplexF>(blocks, p_dst, p_src)); // maybe better performance if I start with second one?
    else if (bcopy_accelerator_accelerator<ComplexF>(blocks, p_dst, p_src));
    else if (bcopy_accelerator_accelerator<double>(blocks, p_dst, p_src));
    else if (bcopy_accelerator_accelerator<float>(blocks, p_dst, p_src));
    else {
      ERR("No fast copy method for block size %ld accelerator<>accelerator implemented", (long)blocks.first);
    }
  } else {
    ERR("Unknown memory copy pattern");
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

  // if there is no buffer, directly issue separate isend / irecv for each block
  if (comm_buffers_type == mt_none) {

    // first start remote submissions
    for (auto & ranks : blocks) {
      rank_t dst_rank = ranks.first.first;
      rank_t src_rank = ranks.first.second;

      if (src_rank == this->rank && dst_rank != this->rank) {
	for (auto & indices : ranks.second) {
	  index_t dst_idx = indices.first.first;
	  index_t src_idx = indices.first.second;

	  for (auto & block : indices.second.second) {
	    this->isend(dst_rank, (char*)base_src[src_idx].ptr + block.start_src, indices.second.first);
	  }
	}
      } else if (src_rank != this->rank && dst_rank == this->rank) {
	for (auto & indices : ranks.second) {
	  index_t dst_idx = indices.first.first;
	  index_t src_idx = indices.first.second;

	  for (auto & block : indices.second.second) {
	    this->irecv(src_rank, (char*)base_dst[dst_idx].ptr + block.start_dst, indices.second.first);
	  }
	}
      }
    }

  } else {

    // if there is a buffer, first gather in communication buffer
    for (auto & ranks : send_blocks) {
      rank_t dst_rank = ranks.first;
      auto & dst = send_buffers.at(dst_rank).view;

      for (auto & indices : ranks.second) {
	index_t src_idx = indices.first;
	bcopy(indices.second, dst, base_src[src_idx]);
      }
    }

    // send/recv buffers
    for (auto & buf : send_buffers) {
      this->isend(buf.first, buf.second.view.ptr, buf.second.view.sz);
    }
    for (auto & buf : recv_buffers) {
      this->irecv(buf.first, buf.second.view.ptr, buf.second.view.sz);
    }
  }


  // then do local copies
 
  for (auto & ranks : blocks) {
    rank_t dst_rank = ranks.first.first;
    rank_t src_rank = ranks.first.second;
    
    if (src_rank == this->rank && dst_rank == this->rank) {
      for (auto & indices : ranks.second) {
	index_t dst_idx = indices.first.first;
	index_t src_idx = indices.first.second;
	
	bcopy(indices.second,base_dst[dst_idx],base_src[src_idx]);
      }
    }
  }
  
  // then wait for remote copies to finish
  this->waitall();

  // if buffer was used, need to re-distribute locally
  if (comm_buffers_type != mt_none) {
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
	bcopy(indices.second, base_dst[dst_idx], src);
      }
    }
  }

}

template class global_transfer<int>;
template class global_memory_view<uint64_t,int,uint32_t>;
template class global_memory_transfer<uint64_t,int,uint32_t>;

void test_global_memory_system() {

  int rank = CartesianCommunicator::RankWorld();
  gm_transfer plan(rank, CartesianCommunicator::communicator_world);
  gm_transfer plan_host_buf(rank, CartesianCommunicator::communicator_world);
  int ranks = plan.mpi_ranks;
  //printf("Rank %d/%d here\n",rank,ranks);
  
  gm_view osrc, odst;
  
  size_t word = sizeof(double);
  size_t word_half = word/2;
  size_t nwords = 512;
  size_t nindex = 6;
  
  // every node requests a specific
  srand (time(NULL));
  std::vector<int> src_ranks(nwords*nindex);
  std::vector<int> src_offset(nwords*nindex);
  std::vector<int> src_index(nwords*nindex);
  
  for (int i=0;i<nindex*nwords;i++) {
    src_ranks[i] = rand() % ranks;
    src_offset[i] = rand() % nwords;
    src_index[i] = rand() % nindex;
  }

  //std::cout << GridLogMessage << "Test setup:" << src_ranks << std::endl << src_offset << std::endl << src_index << std::endl;

  osrc.block_size = word_half;
  odst.block_size = word;

  for (int i=0;i<nindex;i++) {
    for (int j=0;j<nwords;j++) {
      int rs = src_ranks[j + i*nwords];
      int js = src_offset[j + i*nwords];
      int is = src_index[j + i*nwords];
      osrc.blocks.push_back( { rs, (uint32_t)is, js*word } ); // rank, index, offset, size
      osrc.blocks.push_back( { rs, (uint32_t)is, js*word + word_half } ); // rank, index, offset, size
      odst.blocks.push_back( { rank, (uint32_t)i, j*word } ); // rank, index, offset, size
    }
  }
  
  plan.create(odst, osrc, mt_none);
  plan_host_buf.create(odst, osrc, mt_host);
  
  // prepare test data and execute
  std::vector< std::vector<double> > host_src(nindex);
  std::vector< std::vector<double> > host_dst(nindex);
  for (int i=0;i<nindex;i++) {
    host_src[i].resize(nwords);
    host_dst[i].resize(nwords);
    for (int j=0;j<nwords;j++)
      host_src[i][j] = rank * 1000 + j + 100000 * i;
  }
  
  std::vector<gm_transfer::memory_view> dst, src;
  for (int i=0;i<nindex;i++) {
    dst.push_back( { mt_host,&host_dst[i][0],nwords*sizeof(double)} );
    src.push_back( { mt_host,&host_src[i][0],nwords*sizeof(double)} );
  }

  for (int iter=0;iter<2;iter++) {

    for (int i=0;i<nindex;i++) {
      for (int j=0;j<nwords;j++) {
	host_dst[i][j] = -0.1;
      }
    }
    
    if (iter == 0)
      plan.execute(dst,src);
    else
      plan_host_buf.execute(dst,src);

    // test
    for (int i=0;i<nindex;i++) {
      for (int j=0;j<nwords;j++) {
	int rs = src_ranks[j + i*nwords];
	int js = src_offset[j + i*nwords];
	int is = src_index[j + i*nwords];
	
	double expected = rs * 1000 + js + 100000 * is;
	double have = host_dst[i][j];
	if (have != expected) {
	  printf("ITER%d, Rank %d has an error %g != %g (%d %d)\n",iter,rank,expected,have,
		 i,j);
	}
      }
    }
  }

}
