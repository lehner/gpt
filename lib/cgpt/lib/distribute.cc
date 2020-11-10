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
  printf("WAIT %d\n",(int)requests.size());
  std::vector<MPI_Status> stat(requests.size());
  ASSERT(MPI_SUCCESS == MPI_Waitall((int)requests.size(), &requests[0], &stat[0]));
  requests.clear();
#endif
}

template<typename rank_t>
void global_transfer<rank_t>::isend(rank_t other_rank, const void* pdata, size_t sz) {
  if (sz <= size_mpi_max) {
#ifdef CGPT_USE_MPI
    printf("Send from %d to %d, %d bytes from %p (%g double)\n",this->rank,other_rank,(int)sz,pdata,*(double*)pdata);
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
    printf("Recv from %d to %d, %d bytes to %p\n",other_rank,this->rank,(int)sz,pdata);
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
  for (auto r : ranks_from_which_I_will_receive_data) {
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
global_memory_view<offset_t,rank_t,index_t> global_memory_view<offset_t,rank_t,index_t>::merged() const {

  global_memory_view<offset_t,rank_t,index_t> ret;
  if (blocks.size()) {
    ret.blocks.push_back(blocks[0]);
    size_t c = 0;
    
    for (size_t i=1;i<blocks.size();i++) {
      auto & bc = ret.blocks[c];
      auto & bi = blocks[i];
      if (bc.rank == bi.rank && bc.index == bi.index &&
	  bi.start == (bc.start + bc.size)) {
	bc.size += bi.size;
      } else {
	c = i;
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
  struct {
    bool operator()(const block_t& a, const block_t& b) const
    {
      return a.start_src < b.start_src;
    }
  } less;
  
  for (auto & ranks : blocks) {
    for (auto & indices : ranks.second) {
      auto & blocks = indices.second;
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
	  c = i;
	  ret.push_back(bi);
	}
      }

      blocks = ret;
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

  // then prepare packets for each node
  print();
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::create_comm_buffers(memory_type mt) {

  // first remove existing buffers
  send_buffers.clear();
  recv_buffers.clear();
  comm_buffers_type = mt;

  // then exit if nothing to create
  if (mt == mt_none)
    return;

  // Need to compute size of communications buffer for each target node
  std::map<rank_t, size_t> send_size, recv_size;
  for (auto ranks : blocks) {
    rank_t dst_rank = ranks.first.first;
    rank_t src_rank = ranks.first.second;

    if (dst_rank == src_rank) {
      ASSERT(src_rank == this->rank);
      // no communication
      continue;
    }

    size_t sz = 0;
    for (auto indices : ranks.second) {
      for (auto block : indices.second) {      
	sz += block.size;
      }
    }

    if (dst_rank == this->rank) {
      recv_size[src_rank] += sz;
    } else if (src_rank == this->rank) {
      send_size[dst_rank] += sz;
    } else {
      ERR("Mismatched comm info at rank %ld",this->rank);
    }
  }

  // allocate buffers
  for (auto s : send_size) {
    printf("Rank %d has a send_buffer of size %d for rank %d\n",
	   this->rank, (int)s.second, (int)s.first);
    send_buffers.insert(std::make_pair(s.first,memory_buffer(s.second, mt)));
  }

  for (auto s : recv_size) {
    printf("Rank %d has a recv_buffer of size %d for rank %d\n",
	   this->rank, (int)s.second, (int)s.first);
    recv_buffers.insert(std::make_pair(s.first,memory_buffer(s.second, mt)));
  }
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::print() {

  for (auto ranks : blocks) {
    for (auto indices : ranks.second) {
      std::cout << GridLogMessage 
		<< "[" << ranks.first.first << "," << ranks.first.second << "]"
		<< "[" << indices.first.first << "," << indices.first.second << "] : " << std::endl;

      for (auto block : indices.second) {
	std::cout << GridLogMessage << block.start_dst << " <- " <<
	  block.start_src << " for " << block.size << std::endl;
      }
    }
  }
}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::bcopy(const std::vector<block_t>& blocks,
							    std::pair<memory_type,void*>& base_dst, 
							    const std::pair<memory_type,void*>& base_src) {
  
  memory_type mt_dst = base_dst.first;
  char* p_dst = (char*)base_dst.second;

  memory_type mt_src = base_src.first;
  const char* p_src = (const char*)base_dst.second;
  
  if (mt_dst == mt_host && mt_src == mt_host) {
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
    ERR("Not allowed for now, implement a accelerator_for loop for all of them");
  } else {
    ERR("Unknown memory copy pattern");
  }

}

template<typename offset_t, typename rank_t, typename index_t>
void global_memory_transfer<offset_t,rank_t,index_t>::execute(std::vector<std::pair<memory_type,void*>>& base_dst, 
							      std::vector<std::pair<memory_type,void*>>& base_src) {

  // if there is no buffer, directly issue separate isend / irecv for each block
  if (comm_buffers_type == mt_none) {
    // first start remote submissions
    for (auto ranks : blocks) {
      rank_t dst_rank = ranks.first.first;
      rank_t src_rank = ranks.first.second;

      if (src_rank == this->rank && dst_rank != this->rank) {
	for (auto indices : ranks.second) {
	  index_t dst_idx = indices.first.first;
	  index_t src_idx = indices.first.second;

	  for (auto block : indices.second) {
	    this->isend(dst_rank, (char*)base_src[src_idx].second + block.start_src, block.size);
	  }
	}
      } else if (src_rank != this->rank && dst_rank == this->rank) {
	for (auto indices : ranks.second) {
	  index_t dst_idx = indices.first.first;
	  index_t src_idx = indices.first.second;

	  for (auto block : indices.second) {
	    this->irecv(src_rank, (char*)base_dst[dst_idx].second + block.start_dst, block.size);
	  }
	}
      }
    }

    // then do local copies
    for (auto ranks : blocks) {
      rank_t dst_rank = ranks.first.first;
      rank_t src_rank = ranks.first.second;
      
      if (src_rank == this->rank && dst_rank == this->rank) {
	for (auto indices : ranks.second) {
	  index_t dst_idx = indices.first.first;
	  index_t src_idx = indices.first.second;

	  bcopy(indices.second,base_dst[dst_idx],base_src[src_idx]);
	}
      }
    }

    // then wait for remote copies to finish
    this->waitall();

  } else {
    ERR("Not yet implemented");
  }

  // if there is a buffer, first gather in communication buffer
}


/*
  ASSERT(offset_dst.size() == offset_src.size());

  // gather all rank_src -> rank_dst requests from all mpi ranks

  for (auto & r2r : local_r2r_plan) {
    bcopy_plan::merge(r2r.second);
  }

  // now sync between all ranks
  //std::vector<memory_offset_t> 
  //long size = wishlist.size();
  //std::vector<long> rank_size(mpi_ranks);

  //
  


  // root should know all of my r1_r2_size
  // then root lets all nodes know

  // this creates the recv_plan and send_plan

  // and also the local copy plan
  */


template class global_memory_view<uint64_t,int,uint32_t>;
template class global_memory_transfer<uint64_t,int,uint32_t>;


// legacy below

cgpt_distribute::cgpt_distribute(int _rank, Grid_MPI_Comm _comm) : rank(_rank), comm(_comm) {

#ifdef CGPT_USE_MPI
  //MPI_COMM_WORLD
  MPI_Comm_size(comm,&mpi_ranks);
  MPI_Comm_rank(comm,&mpi_rank);
  mpi_rank_map.resize(mpi_ranks,0);
  ASSERT(rank < mpi_ranks);
  mpi_rank_map[rank] = mpi_rank;
  ASSERT(MPI_SUCCESS == MPI_Allreduce(MPI_IN_PLACE,&mpi_rank_map[0],mpi_ranks,MPI_INT,MPI_SUM,comm));
#else
  mpi_ranks=1;
  mpi_rank=0;
  mpi_rank_map.push_back(0);
#endif
}

void cgpt_distribute::split(const std::vector<coor>& c, std::map<int,mp>& s) const {
  long dst = 0;
  for (auto& f : c) {
    auto & t = s[f.rank];
    t.src.push_back(f.offset);
    t.dst.push_back(dst++);
  }
}

void cgpt_distribute::create_plan(const std::vector<coor>& c, plan& p) const {

  // split mapping by rank
  split(c,p.cr);

  // head node needs to learn all the remote requirements
  std::vector<long> wishlist;
  packet_prepare_need(wishlist,p.cr);

  // head node collects the wishlists
  std::map< int, std::vector<long> > wishlists;
  wishlists_to_root(wishlist,wishlists);

  // now root tells every node which other nodes needs how much of its data
  send_tasks_to_ranks(wishlists,p.tasks);
}

void cgpt_distribute::copy_to(const plan& p, std::vector<data_simd>& src, void* dst) {
  
  // copy local data
  const auto& ld = p.cr.find(rank);
  if (ld != p.cr.cend())
    copy_data(ld->second,src,dst);

  // receive the requested wishlist from my task ranks
  copy_remote(p.tasks,p.cr,src,dst);

}

void cgpt_distribute::copy_from(const plan& p, void* src, long src_size, std::vector<data_simd>& dst) {

  // copy local data
  const auto& ld = p.cr.find(rank);
  if (ld != p.cr.cend())
    copy_data_rev(ld->second,src,src_size,dst);

  // receive the requested wishlist from my task ranks
  copy_remote_rev(p.tasks,p.cr,src,src_size,dst);

}

void cgpt_distribute::copy(const plan& p_dst, const plan& p_src, std::vector<data_simd>& dst, std::vector<data_simd>& src) {

  // for now only support local data 
  ASSERT(p_dst.tasks.size() == 0);
  ASSERT(p_src.tasks.size() == 0);
  ASSERT(dst.size() == src.size());

  //double t0 = cgpt_time();

  // local->local
  const auto& ld_src = p_src.cr.find(rank);
  const auto& ld_dst = p_dst.cr.find(rank);
  if ((ld_src != p_src.cr.cend()) && (ld_dst != p_dst.cr.end())) {
    auto & _dst = ld_dst->second.src;
    auto & _src = ld_src->second.src;
    ASSERT(_dst.size() == _src.size());
    long len = _dst.size();

    for (long ni = 0; ni < dst.size(); ni++) {
      
      // src
      auto & src_offset_data = src[ni].offset_data;
      long src_N_ni = src_offset_data.size();
      long src_Nsimd = src[ni].Nsimd;
      long src_simd_word = src[ni].simd_word;
      long src_word = src[ni].word;
      long src_si_stride = src_Nsimd * src_simd_word;
      long src_o_stride = src_Nsimd * src_word;
      char* __src = (char*)src[ni].local;

      // dst
      auto & dst_offset_data = dst[ni].offset_data;
      long dst_N_ni = dst_offset_data.size();
      long dst_Nsimd = dst[ni].Nsimd;
      long dst_simd_word = dst[ni].simd_word;
      long dst_word = dst[ni].word;
      long dst_si_stride = dst_Nsimd * dst_simd_word;
      long dst_o_stride = dst_Nsimd * dst_word;
      char* __dst = (char*)dst[ni].local;

      // eq
      ASSERT(dst_N_ni == src_N_ni);
      ASSERT(dst_simd_word == src_simd_word);
      long N_ni = dst_N_ni;
      long simd_word = dst_simd_word;

      ASSERT(len % dst_Nsimd == 0);
      long len_simd = len / dst_Nsimd;

      //double t0 =cgpt_time(); 
      thread_region
	{

	  long copy_src_start = -1;
	  long copy_dst_start = -1;
	  long copy_size = 0;

	  thread_for_in_region(idx_simd, len_simd, {
	  
	      for (long si = 0; si < N_ni; si++) {

		for (long idx_internal = 0; idx_internal < dst_Nsimd; idx_internal++) {

		  long idx = idx_simd * dst_Nsimd + idx_internal;

		  long offset_src = _src[idx];
		  long _odx_src = offset_src / SIMD_BASE;
		  long _idx_src = offset_src % SIMD_BASE;
		  
		  long offset_dst = _dst[idx];
		  long _odx_dst = offset_dst / SIMD_BASE;
		  long _idx_dst = offset_dst % SIMD_BASE;
		  
		  long src_idx = src_si_stride*src_offset_data[si] + simd_word*_idx_src + src_o_stride*_odx_src;
		  long dst_idx = dst_si_stride*dst_offset_data[si] + simd_word*_idx_src + dst_o_stride*_odx_src;
		  
		  if (copy_src_start + copy_size == src_idx &&
		      copy_dst_start + copy_size == dst_idx) {
		    copy_size += simd_word;
		  } else {
		    
		    if (copy_src_start != -1) {
		      // perform copy of previous block
		      //if (thread_num() == 0)
		      //std::cout << GridLogMessage << copy_dst_start << " <- " << copy_src_start << " size " << copy_size << "( odx = " << _odx_dst << ", idx = " << _idx_dst << ")" << std::endl;
		      memcpy(&__dst[copy_dst_start],&__src[copy_src_start],copy_size);
		      //if (copy_size == 64) {
		      //  vComplexD* a = (vComplexD*)&__dst[copy_dst_start];
		      //  vComplexD* b = (vComplexD*)&__src[copy_src_start];
		      //  *a = *b;
		      //}
		    }
		    
		    copy_src_start = src_idx;
		    copy_dst_start = dst_idx;
		    copy_size = simd_word;
		  }
		}
	      }
		  

	    });

	  if (copy_src_start != -1) {
	    // perform copy of block
	    //if (thread_num() == 0)
	    //  std::cout << GridLogMessage << copy_dst_start << " <- " << copy_src_start << " size " << copy_size << std::endl;
	    memcpy(&__dst[copy_dst_start],&__src[copy_src_start],copy_size);
	  }
	}
    }

    //double t1 = cgpt_time();
    //std::cout << GridLogMessage << "Copy " << t1 - t0 << std::endl;
    //for (long i =0;i<ld_src->second.dst.size();i++) {
    //  std::cout << GridLogMessage << ld_src->second.dst[i] << " <- " << ld_dst->second.dst[i] << std::endl;
    //}
    return;
  }

  ERR("Invalid copy setup");
}

long cgpt_distribute::word_total(std::vector<data_simd>& src) {
  long r = 0;
  for (auto& s : src) {
    ASSERT(s.Nsimd <= SIMD_BASE);
    r+=s.simd_word * s.offset_data.size();
  }
  return r;
}

void cgpt_distribute::copy_remote(const std::vector<long>& tasks, const std::map<int,mp>& cr,
				  std::vector<data_simd>& _src, void* _dst) {
#ifdef CGPT_USE_MPI
  assert(tasks.size() % 2 == 0);
  std::vector<MPI_Request> req;
  std::map<int, std::vector<long> > remote_needs_offsets;
  long _word_total = word_total(_src);

  for (int i = 0; i < tasks.size() / 2; i++) {
    int dest_rank  = (int)tasks[2*i + 0];
    long dest_n    = tasks[2*i + 1];

    auto & w = remote_needs_offsets[dest_rank];

    w.resize(dest_n);

    MPI_Request r;
    ASSERT(MPI_SUCCESS == MPI_Irecv(&w[0],dest_n,MPI_LONG,mpi_rank_map[dest_rank],0x3,comm,&r));
    req.push_back(r);

  }
  for (auto & f : cr) {
    int dest_rank = f.first;
    if (dest_rank != rank) {
      auto & off = f.second.src;
      MPI_Request r;
      ASSERT(MPI_SUCCESS == MPI_Isend(&off[0],off.size(),MPI_LONG,mpi_rank_map[dest_rank],0x3,comm,&r));
      req.push_back(r);
    }
  }

  {
    std::vector<MPI_Status> stat(req.size());
    ASSERT(MPI_SUCCESS == MPI_Waitall((int)req.size(), &req[0], &stat[0]));
  }

  // now send all data
  req.resize(0);
  std::map<int, std::vector<char> > buf;
  for (int i = 0; i < tasks.size() / 2; i++) {
    int dest_rank  = (int)tasks[2*i + 0];
    long dest_n    = tasks[2*i + 1];

    auto & w = buf[dest_rank];
    auto & n = remote_needs_offsets[dest_rank];
    w.resize(dest_n * _word_total);
    char* dst = (char*)&w[0];

    {
      thread_for(idx, dest_n, {
	  long offset = n[idx];
	  long _odx = offset / SIMD_BASE;
	  long _idx = offset % SIMD_BASE;

	  for (long ni = 0; ni < _src.size(); ni++) {
	    char* src = (char*)_src[ni].local;
	    long Nsimd = _src[ni].Nsimd;
	    long simd_word = _src[ni].simd_word;
	    long word = _src[ni].word;
	    auto & offset_data = _src[ni].offset_data;
	    auto & offset_buffer = _src[ni].offset_buffer;
	    long N_ni = offset_data.size();
	    long si_stride = Nsimd * simd_word;
	    long o_stride = Nsimd * word;

	    for (long si = 0; si < N_ni; si++) {
	      memcpy(&dst[_word_total*idx + offset_buffer[si]*simd_word],&src[si_stride*offset_data[si] + simd_word*_idx + o_stride*_odx],simd_word);
	    }
	  }
	});
    }

    MPI_Request r;
    ASSERT(w.size() < INT_MAX);
    ASSERT(MPI_SUCCESS == MPI_Isend(&w[0],(int)w.size(),MPI_CHAR,mpi_rank_map[dest_rank],0x2,comm,&r));
    req.push_back(r);
  }

  // receive data
  std::map<int, std::vector<char> > bufr;
  for (auto & f : cr) {
    int dest_rank = f.first;
    if (dest_rank != rank) {
      long dest_n = f.second.src.size();
      auto & w = bufr[dest_rank];
      w.resize(dest_n * _word_total);
      MPI_Request r;
      ASSERT(w.size() < INT_MAX);
      ASSERT(MPI_SUCCESS == MPI_Irecv(&w[0],(int)w.size(),MPI_CHAR,mpi_rank_map[dest_rank],0x2,comm,&r));
      req.push_back(r);
    }
  }
  {
    std::vector<MPI_Status> stat(req.size());
    ASSERT(MPI_SUCCESS == MPI_Waitall((int)req.size(), &req[0], &stat[0]));
  }

  // now bufr contains all data to copy to dest
  char* dst = (char*)_dst;
  for (auto & f : cr) {
    int dest_rank = f.first;
    if (dest_rank != rank) {
      auto & d = f.second.dst;
      auto & w = bufr[dest_rank];
      long dest_n = f.second.src.size();
      assert(dest_n == d.size());
      assert(dest_n*_word_total == w.size());
      thread_for(idx, dest_n,{
	  memcpy(&dst[_word_total*d[idx]],&w[idx*_word_total],_word_total);
	});
    }
  }  
#endif
}

void cgpt_distribute::copy_remote_rev(const std::vector<long>& tasks, const std::map<int,mp>& cr,
				      void* _src, long _src_size, std::vector<data_simd>& _dst) {
#ifdef CGPT_USE_MPI
  /*
    tasks here are requests from other nodes to set local data

    cr are the local requests to set remote data
  */

  long _word_total = word_total(_dst);
  assert(tasks.size() % 2 == 0);
  std::vector<MPI_Request> req;
  std::map<int, std::vector<long> > remote_needs_offsets; // we will gather here the offsets that the remote wants to set with the sent buffer
  for (int i = 0; i < tasks.size() / 2; i++) {
    int dest_rank  = (int)tasks[2*i + 0];
    long dest_n    = tasks[2*i + 1];

    auto & w = remote_needs_offsets[dest_rank];

    w.resize(dest_n);

    MPI_Request r;
    ASSERT(MPI_SUCCESS == MPI_Irecv(&w[0],dest_n,MPI_LONG,mpi_rank_map[dest_rank],0x3,comm,&r));
    req.push_back(r);

  }
  for (auto & f : cr) {
    int dest_rank = f.first;
    if (dest_rank != rank) {
      auto & off = f.second.src; // OK
      MPI_Request r;
      ASSERT(MPI_SUCCESS == MPI_Isend(&off[0],off.size(),MPI_LONG,mpi_rank_map[dest_rank],0x3,comm,&r));
      req.push_back(r);
    }
  }

  {
    std::vector<MPI_Status> stat(req.size());
    ASSERT(MPI_SUCCESS == MPI_Waitall((int)req.size(), &req[0], &stat[0]));
  }

  // now send all data
  req.resize(0);
  std::map<int, std::vector<char> > buf;
  for (auto & f : cr) {
    int dest_rank  = f.first;
    if (dest_rank != rank) {
      auto & w = buf[dest_rank];
      auto & n = f.second.dst; // source positions to be sent to dest_rank
      long dest_n = (long)n.size();

      w.resize(dest_n * _word_total);
      char* dst = (char*)&w[0];
      char* src = (char*)_src;

      thread_for(idx, dest_n,{
	  long cur_pos = 0;
	  while (cur_pos < _word_total) {
	    long dst_offset = idx*_word_total + cur_pos;
	    long src_offset = (_word_total*n[idx] + cur_pos) % _src_size;
	    long block_size = std::min(_word_total - cur_pos,_src_size - src_offset);
	    memcpy(dst + dst_offset,src + src_offset,block_size);
	    cur_pos += block_size;
	  }
	});
      
      MPI_Request r;
      size_t t_left = w.size();
      size_t t_max = INT_MAX;
      char* t_ptr = &w[0];
      while (t_left) {
	size_t t_size = std::min(t_left,t_max);
	ASSERT(MPI_SUCCESS == MPI_Isend(t_ptr,t_size,MPI_CHAR,mpi_rank_map[dest_rank],0x2,comm,&r));
	req.push_back(r);
	t_ptr += t_size;
	t_left -= t_size;
      }
    }
  }

  // receive data
  std::map<int, std::vector<char> > bufr;
  for (int i = 0; i < tasks.size() / 2; i++) {
    int dest_rank  = (int)tasks[2*i + 0];
    long dest_n    = tasks[2*i + 1];

    auto & w = bufr[dest_rank];
    w.resize(dest_n * _word_total);
    
    MPI_Request r;
    size_t t_left = w.size();
    size_t t_max = INT_MAX;
    char* t_ptr = &w[0];
    while (t_left) {
      size_t t_size = std::min(t_left,t_max);
      ASSERT(MPI_SUCCESS == MPI_Irecv(t_ptr,(int)t_size,MPI_CHAR,mpi_rank_map[dest_rank],0x2,comm,&r));
      req.push_back(r);
      t_ptr += t_size;
      t_left -= t_size;
    }
  }
  {
    std::vector<MPI_Status> stat(req.size());
    ASSERT(MPI_SUCCESS == MPI_Waitall((int)req.size(), &req[0], &stat[0]));
  }

  // now bufr contains all data to copy to local
  for (int i = 0; i < tasks.size() / 2; i++) {
    int dest_rank  = (int)tasks[2*i + 0];
    long dest_n    = tasks[2*i + 1];

    auto & s = remote_needs_offsets[dest_rank]; // offsets the remote wants to set
    auto & w = bufr[dest_rank];
    assert(dest_n == s.size());
    assert(dest_n*_word_total == w.size());
    
    {
      char* src = (char*)&w[0];
      
      thread_for(idx, dest_n, {
	  long offset = s[idx];
	  long _odx = offset / SIMD_BASE;
	  long _idx = offset % SIMD_BASE;

	  for (long ni = 0; ni < _dst.size(); ni++) {
	    char* dst = (char*)_dst[ni].local;
	    long Nsimd = _dst[ni].Nsimd;
	    long simd_word = _dst[ni].simd_word;
	    long word = _dst[ni].word;
	    auto & offset_data = _dst[ni].offset_data;
	    auto & offset_buffer = _dst[ni].offset_buffer;
	    long N_ni = offset_data.size();
	    long si_stride = Nsimd * simd_word;
	    long o_stride = Nsimd * word;

	    for (long si = 0; si < N_ni; si++) {
	      memcpy(&dst[si_stride*offset_data[si] + simd_word*_idx + o_stride*_odx],&src[_word_total*idx + offset_buffer[si]*simd_word],simd_word);
	    }
	  }
	});
    }
  }  
#endif
}

void cgpt_distribute::copy_data(const mp& m, std::vector<data_simd>& _src, void* _dst) {
  long len = m.src.size();
  unsigned char* dst = (unsigned char*)_dst;
  long _word_total = word_total(_src);

  //double t0 =cgpt_time();
  thread_for(idx, len, {
      long offset = m.src[idx];
      long _odx = offset / SIMD_BASE;
      long _idx = offset % SIMD_BASE;
      
      for (long ni = 0; ni < _src.size(); ni++) {
	char* src = (char*)_src[ni].local;
	long Nsimd = _src[ni].Nsimd;
	long simd_word = _src[ni].simd_word;
	long word = _src[ni].word;
	auto & offset_data = _src[ni].offset_data;
	auto & offset_buffer = _src[ni].offset_buffer;
	long N_ni = offset_data.size();
	long si_stride = Nsimd * simd_word;
	long o_stride = Nsimd * word;
	
	for (long si = 0; si < N_ni; si++) {
	  memcpy(&dst[_word_total*m.dst[idx] + offset_buffer[si]*simd_word],&src[si_stride*offset_data[si] + simd_word*_idx + o_stride*_odx],simd_word);
	}

      }
    });

  //double t1=cgpt_time();
  //double GB = _src[0].word * _src.size() * len / 1024. / 1024. /1024.;
  //printf("%g GB/s %g\n",2*GB/(t1-t0),t1-t0);
}

void cgpt_distribute::copy_data_rev(const mp& m, void* _src, long _src_size, std::vector<data_simd>& _dst) {
  long len = m.src.size();
  unsigned char* src = (unsigned char*)_src;
  long _word_total = word_total(_dst);

  //double t0 =cgpt_time(); 
  thread_for(idx, len, {
      long offset = m.src[idx];
      long _odx = offset / SIMD_BASE;
      long _idx = offset % SIMD_BASE;
      
      for (long ni = 0; ni < _dst.size(); ni++) {
	char* dst = (char*)_dst[ni].local;
	long Nsimd = _dst[ni].Nsimd;
	long simd_word = _dst[ni].simd_word;
	long word = _dst[ni].word;
	auto & offset_data = _dst[ni].offset_data;
	auto & offset_buffer = _dst[ni].offset_buffer;
	long N_ni = offset_data.size();
	long si_stride = Nsimd * simd_word;
	long o_stride = Nsimd * word;
	
	for (long si = 0; si < N_ni; si++) {
	  memcpy(&dst[si_stride*offset_data[si] + simd_word*_idx + o_stride*_odx],&src[(_word_total*m.dst[idx] + offset_buffer[si]*simd_word) % _src_size],simd_word);
	}
      }
    });


  //double t1=cgpt_time();
  //double GB = _dst[0].word * _dst.size() * len / 1024. / 1024. /1024.;
  //printf("rev %g GB/s %g\n",2*GB/(t1-t0),t1-t0);
}

void cgpt_distribute::get_send_tasks_for_rank(int i, const std::map<int, std::vector<long> >& wishlists, std::vector<long>& tasks) const {
  // first find all tasks
  tasks.resize(0);
  for (auto & w : wishlists) {
    auto & d = w.second;
    assert(d.size() % 2 == 0);
    for (size_t j = 0; j < d.size() / 2; j++) {
      long rank = d[2*j + 0];
      long n    = d[2*j + 1];
      if (rank == i) {
	tasks.push_back(w.first);
	tasks.push_back(n);
      }	  
    }
  }
}

void cgpt_distribute::send_tasks_to_ranks(const std::map<int, std::vector<long> >& wishlists, std::vector<long>& tasks) const {
#ifdef CGPT_USE_MPI  
  std::vector<long> lens(mpi_ranks);
  long len;

  if (!rank) {

    std::map< int, std::vector<long> > rank_tasks;
    for (int i=1;i<mpi_ranks;i++) {
      get_send_tasks_for_rank(i,wishlists,rank_tasks[i]);
    }

    // vector of lengths
    for (int i=1;i<mpi_ranks;i++)
      lens[mpi_rank_map[i]] = rank_tasks[i].size();

    ASSERT(MPI_SUCCESS == MPI_Scatter(&lens[0], 1, MPI_LONG, &len, 1, MPI_LONG, mpi_rank_map[0], comm));

    std::vector<MPI_Request> req;
    for (int i=1;i<mpi_ranks;i++) {
      long li = lens[mpi_rank_map[i]];
      if (li != 0) {
	MPI_Request r;
	ASSERT(MPI_SUCCESS == MPI_Isend(&rank_tasks[i][0], li, MPI_LONG,mpi_rank_map[i],0x5,comm,&r));
	req.push_back(r);
      }
    }

    if (req.size() != 0) {
      std::vector<MPI_Status> stat(req.size());
      ASSERT(MPI_SUCCESS == MPI_Waitall((int)req.size(), &req[0], &stat[0]));
    }

    get_send_tasks_for_rank(0,wishlists,tasks); // overwrite mine with my tasks
  } else {

    ASSERT(MPI_SUCCESS == MPI_Scatter(&lens[0], 1, MPI_LONG, &len, 1, MPI_LONG, mpi_rank_map[0], comm));
    // receive
    tasks.resize(len);
    MPI_Status status;
    if (len != 0)
      ASSERT(MPI_SUCCESS == MPI_Recv(&tasks[0],len,MPI_LONG,mpi_rank_map[0],0x5,comm,&status));
  }
#else
  get_send_tasks_for_rank(0,wishlists,tasks); // overwrite mine with my tasks
#endif
}

void cgpt_distribute::wishlists_to_root(const std::vector<long>& wishlist, std::map<int, std::vector<long> >& wishlists) const {
#ifdef CGPT_USE_MPI
  long size = wishlist.size();
  std::vector<long> rank_size(mpi_ranks);

  ASSERT(MPI_SUCCESS == MPI_Gather(&size, 1, MPI_LONG, &rank_size[0], 1, MPI_LONG, mpi_rank_map[0], comm));

  if (rank != 0) {
    if (size)
      ASSERT(MPI_SUCCESS == MPI_Send(&wishlist[0], size, MPI_LONG,mpi_rank_map[0],0x4,comm));
  } else {
    wishlists[0] = wishlist; // my own wishlist can stay

    std::vector<MPI_Request> req;
    for (int i=1;i<mpi_ranks;i++) { // gather wishes from all others
      long rs = rank_size[mpi_rank_map[i]];

      if (rs != 0) {
	auto & w = wishlists[i];
	w.resize(rs);

	if (rs != 0) {
	  MPI_Request r;
	  ASSERT(MPI_SUCCESS == MPI_Irecv(&w[0],rs,MPI_LONG,mpi_rank_map[i],0x4,comm,&r));
	  req.push_back(r);
	}
      }
    }

    if (req.size() != 0) {
      std::vector<MPI_Status> stat(req.size());
      ASSERT(MPI_SUCCESS == MPI_Waitall((int)req.size(), &req[0], &stat[0]));
    }
  }
#endif
}

void cgpt_distribute::packet_prepare_need(std::vector<long>& data, const std::map<int,mp>& cr) const {
  for (auto & f : cr) {
    if (f.first != rank) {
      data.push_back(f.first); // rank
      data.push_back(f.second.src.size());
    }
  }
}
