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
#include <set>

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
    std::cout << " [" << i << "/" << blocks.size() << "] = { " << bc.rank << ", " << bc.index << ", " << bc.start << ", " << bc.size << " }" << std::endl;
  }
}

template<typename offset_t, typename rank_t, typename index_t>
offset_t global_memory_view<offset_t,rank_t,index_t>::size() const {
  offset_t sz = 0;
  for (size_t i=0;i<blocks.size();i++) {
    auto & bc = blocks[i];
    sz += bc.size;
  }
  return sz;
}

template<typename offset_t, typename rank_t, typename index_t>
global_memory_view<offset_t,rank_t,index_t> global_memory_view<offset_t,rank_t,index_t>::merged() const {

  global_memory_view<offset_t,rank_t,index_t> ret;
  if (blocks.size()) {
    if (blocks[0].size)
      ret.blocks.push_back(blocks[0]);
    size_t c = 0;
    
    for (size_t i=1;i<blocks.size();i++) {
      auto & bc = ret.blocks[c];
      auto & bi = blocks[i];
      if (bc.rank == bi.rank && bc.index == bi.index &&
	  bi.start == (bc.start + bc.size)) {
	bc.size += bi.size;
      } else {
	c++;
	if (bi.size)
	  ret.blocks.push_back(bi);
      }
    }
  }

  return ret;
}

template<typename offset_t, typename rank_t, typename index_t>
global_memory_transfer<offset_t,rank_t,index_t>::global_memory_transfer(rank_t _rank, Grid_MPI_Comm _comm) :
  global_transfer<rank_t>(_rank, _comm), comm_buffers_type(mt_none) {
}


template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::fill_blocks_from_view_pair(const view_t& _dst,
										 const view_t& _src) {
  // create local src -> dst command
  auto dst = _dst.merged();
  auto src = _src.merged();
  //auto dst = _dst;
  //auto src = _src;

  size_t is = 0, id = 0;
  while (is < src.blocks.size() &&
	 id < dst.blocks.size()) {
    auto & s = src.blocks[is];
    auto & d = dst.blocks[id];

    offset_t sz = std::min(s.size, d.size);

    blocks
      [std::make_pair(d.rank, s.rank)]
      [std::make_pair(d.index, s.index)]
      .push_back({d.start,s.start,sz});

    if (sz == s.size) {
      is++;
    } else {
      s.start += sz;
      s.size -= sz;
    }

    if (sz == d.size) {
      id++;
    } else {
      d.start += sz;
      d.size -= sz;
    }
  }

  // expect both to be of same size
  ASSERT(is == src.blocks.size() &&
	 id == dst.blocks.size());
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
      std::pair< std::pair<rank_t,rank_t>, std::map< std::pair<index_t,index_t>, std::vector<block_t> > > ranks;
      tasks.second.get(ranks);

      // and merge with current blocks
      for (auto & idx : ranks.second) {
	auto & a = blocks[ranks.first][idx.first];
	auto & b = idx.second;
	a.insert(a.end(), b.begin(), b.end());
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
void global_memory_transfer<offset_t,rank_t,index_t>::optimize() {
  for (auto & ranks : blocks) {
    for (auto & indices : ranks.second) {
      optimize(indices.second);
    }
  }
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::optimize(std::vector<block_t>& blocks) {
  struct {
    bool operator()(const block_t& a, const block_t& b) const
    {
      return a.start_dst < b.start_dst; // sort by destination address (better for first write page mapping)
    }
  } less;
  
  std::sort(blocks.begin(), blocks.end(), less);

  // merge neighboring blocks
  std::vector<block_t> ret;
  ret.push_back(blocks[0]);
  size_t c = 0;
  
  for (size_t i=1;i<blocks.size();i++) {
    auto & bc = ret[c];
    auto & bi = blocks[i];
    if (bi.start_src == bc.start_src &&
	bi.start_dst == bc.start_dst &&
	bi.size == bc.size) {
      continue; // remove duplicates (maybe from different ranks)
    } else if (bi.start_src == (bc.start_src + bc.size) &&
	       bi.start_dst == (bc.start_dst + bc.size)) {
      bc.size += bi.size;
    } else {
      c++;
      ret.push_back(bi);
    }
  }
  
  blocks = ret;
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
      for (auto & bi : indices.second) {
	offset_t end_dst = bi.start_dst + bi.size;
	offset_t end_src = bi.start_src + bi.size;
	if (dst_rank == this->rank && end_dst > bounds_dst[dst_idx])
	  bounds_dst[dst_idx] = end_dst;
	if (src_rank == this->rank && end_src > bounds_src[src_idx])
	  bounds_src[src_idx] = end_src;
      }
    }
  }

}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::create(const view_t& _dst,
							     const view_t& _src,
							     memory_type use_comm_buffers_of_type) {

  // reset
  blocks.clear();

  // fill
  fill_blocks_from_view_pair(_dst,_src);

  // optimize blocks obtained from this rank
  optimize();

  // gather my blocks
  gather_my_blocks();

  // optimize blocks after gathering all of my blocks
  optimize();

  // optionally create communication buffers
  create_comm_buffers(use_comm_buffers_of_type);

  // create bounds
  create_bounds();
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
	for (auto & block : indices.second) {      
	  rb.push_back({ block.start_dst, sz, block.size });
	  sz += block.size;
	}
	optimize(rb);
      }
      recv_size[src_rank] += sz;
    } else if (src_rank == this->rank) {
      size_t sz = 0;
      for (auto & indices : ranks.second) {
	auto& sb = send_blocks[dst_rank][indices.first.second]; // src index
	for (auto & block : indices.second) {      
	  sb.push_back({ sz, block.start_src, block.size });
	  sz += block.size;
	}
	optimize(sb);
      }
      send_size[dst_rank] += sz;
    } else {
      ERR("Mismatched comm info at rank %ld",this->rank);
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

      for (auto & block : indices.second) {
	std::cout << GridLogMessage << block.start_dst << " <- " <<
	  block.start_src << " for " << block.size << std::endl;
      }
    }
  }
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::bcopy(const std::vector<block_t>& blocks,
							    memory_view& base_dst, 
							    const memory_view& base_src) {
  
  memory_type mt_dst = base_dst.type;
  char* p_dst = (char*)base_dst.ptr;

  memory_type mt_src = base_src.type;
  const char* p_src = (const char*)base_src.ptr;
  
  if (mt_dst == mt_host && mt_src == mt_host) {
    // TODO
    // strategy:
    // - define nparallel (number of threads)
    // - nparallel_per_block = nparallel / blocks.size()
    //
    for (size_t i=0;i<blocks.size();i++) {
      auto&b=blocks[i];
      memcpy(&p_dst[b.start_dst],&p_src[b.start_src],b.size);
    }
  } else if (mt_dst == mt_host && mt_src == mt_accelerator) {
    for (size_t i=0;i<blocks.size();i++) {
      auto&b=blocks[i];
      acceleratorCopyFromDevice((void*)&p_src[b.start_src],(void*)&p_dst[b.start_dst],b.size);
    }
  } else if (mt_dst == mt_accelerator && mt_src == mt_host) {
    for (size_t i=0;i<blocks.size();i++) {
      auto&b=blocks[i];
      acceleratorCopyToDevice((void*)&p_src[b.start_src],(void*)&p_dst[b.start_dst],b.size);
    }
  } else if (mt_dst == mt_accelerator && mt_src == mt_accelerator) {
    for (size_t i=0;i<blocks.size();i++) {
      auto&b=blocks[i];
      acceleratorCopyDeviceToDevice((void*)&p_src[b.start_src],(void*)&p_dst[b.start_dst],b.size);
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

	  for (auto & block : indices.second) {
	    this->isend(dst_rank, (char*)base_src[src_idx].ptr + block.start_src, block.size);
	  }
	}
      } else if (src_rank != this->rank && dst_rank == this->rank) {
	for (auto & indices : ranks.second) {
	  index_t dst_idx = indices.first.first;
	  index_t src_idx = indices.first.second;

	  for (auto & block : indices.second) {
	    this->irecv(src_rank, (char*)base_dst[dst_idx].ptr + block.start_dst, block.size);
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

  for (int i=0;i<nindex;i++) {
    for (int j=0;j<nwords;j++) {
      int rs = src_ranks[j + i*nwords];
      int js = src_offset[j + i*nwords];
      int is = src_index[j + i*nwords];
      osrc.blocks.push_back( { rs, (uint32_t)is, js*word, word_half } ); // rank, index, offset, size
      osrc.blocks.push_back( { rs, (uint32_t)is, js*word + word_half, word_half } ); // rank, index, offset, size
      odst.blocks.push_back( { rank, (uint32_t)i, j*word, word } ); // rank, index, offset, size
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
