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

#if defined (GRID_COMMS_MPI3)
#define USE_MPI 1
#endif

cgpt_distribute::cgpt_distribute(int _rank, Grid_MPI_Comm _comm) : rank(_rank), comm(_comm) {

#ifdef USE_MPI
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

void cgpt_distribute::split(const std::vector<coor>& c, std::map<int,mp>& s) {
  long dst = 0;
  for (auto& f : c) {
    auto & t = s[f.rank];
    t.src.push_back(f.offset);
    t.dst.push_back(dst++);
  }
}

void cgpt_distribute::create_plan(const std::vector<coor>& c, plan& p) {

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

void cgpt_distribute::copy_to(const plan& p,std::vector<data_simd>& src, void* dst) {
  
  // copy local data
  const auto& ld = p.cr.find(rank);
  if (ld != p.cr.cend())
    copy_data(ld->second,src,dst);

  // receive the requested wishlist from my task ranks
  copy_remote(p.tasks,p.cr,src,dst);

}

void cgpt_distribute::copy_from(const plan& p,void* src,std::vector<data_simd>& dst) {

  // copy local data
  const auto& ld = p.cr.find(rank);
  if (ld != p.cr.cend())
    copy_data_rev(ld->second,src,dst);

  // receive the requested wishlist from my task ranks
  copy_remote_rev(p.tasks,p.cr,src,dst);

}

void cgpt_distribute::copy_remote(const std::vector<long>& tasks, const std::map<int,mp>& cr,std::vector<data_simd>& _src, void* _dst) {
#ifdef USE_MPI
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

	  long si_base = 0;
	  for (long ni = 0; ni < _src.size(); ni++) {
	    char* src = (char*)_src[ni].local;
	    long Nsimd = _src[ni].Nsimd;
	    long simd_word = _src[ni].simd_word;
	    long word = _src[ni].word;
	    long N_ni = word / simd_word;
	    long si_stride = Nsimd * simd_word;
	    long o_stride = Nsimd * word;

	    for (long si = 0; si < N_ni; si++) {
	      memcpy(&dst[_word_total*idx + (si_base + si)*simd_word],&src[si_stride*si + simd_word*_idx + o_stride*_odx],simd_word);
	    }

	    si_base += N_ni;
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

void cgpt_distribute::copy_remote_rev(const std::vector<long>& tasks, const std::map<int,mp>& cr,void* _src,std::vector<data_simd>& _dst) {
#ifdef USE_MPI
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
	  memcpy(&dst[idx*_word_total],&src[_word_total*n[idx]],_word_total);
	});
      
      MPI_Request r;
      ASSERT(w.size() < INT_MAX);
      ASSERT(MPI_SUCCESS == MPI_Isend(&w[0],(int)w.size(),MPI_CHAR,mpi_rank_map[dest_rank],0x2,comm,&r));
      req.push_back(r);
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
    ASSERT(w.size() < INT_MAX);
    ASSERT(MPI_SUCCESS == MPI_Irecv(&w[0],(int)w.size(),MPI_CHAR,mpi_rank_map[dest_rank],0x2,comm,&r));
    req.push_back(r);
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

	  long si_base = 0;
	  for (long ni = 0; ni < _dst.size(); ni++) {
	    char* dst = (char*)_dst[ni].local;
	    long Nsimd = _dst[ni].Nsimd;
	    long simd_word = _dst[ni].simd_word;
	    long word = _dst[ni].word;
	    long N_ni = word / simd_word;
	    long si_stride = Nsimd * simd_word;
	    long o_stride = Nsimd * word;

	    for (long si = 0; si < N_ni; si++) {
	      memcpy(&dst[si_stride*si + simd_word*_idx + o_stride*_odx],&src[_word_total*idx + (si_base + si)*simd_word],simd_word);
	    }
	    
	    si_base += N_ni;
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

  thread_for(idx, len, {
      long offset = m.src[idx];
      long _odx = offset / SIMD_BASE;
      long _idx = offset % SIMD_BASE;

      long si_base = 0;
      for (long ni = 0; ni < _src.size(); ni++) {
	char* src = (char*)_src[ni].local;
	long Nsimd = _src[ni].Nsimd;
	long simd_word = _src[ni].simd_word;
	long word = _src[ni].word;
	long N_ni = word / simd_word;
	long si_stride = Nsimd * simd_word;
	long o_stride = Nsimd * word;

	for (long si = 0; si < N_ni; si++) {
	  memcpy(&dst[_word_total*m.dst[idx] + (si_base + si)*simd_word],&src[si_stride*si + simd_word*_idx + o_stride*_odx],simd_word);
	}

	si_base += N_ni;
      }
    });
}

void cgpt_distribute::copy_data_rev(const mp& m, void* _src, std::vector<data_simd>& _dst) {
  long len = m.src.size();
  unsigned char* src = (unsigned char*)_src;
  long _word_total = word_total(_dst);
 
  thread_for(idx, len, {
      long offset = m.src[idx];
      long _odx = offset / SIMD_BASE;
      long _idx = offset % SIMD_BASE;
      
      long si_base = 0;
      for (long ni = 0; ni < _dst.size(); ni++) {
	char* dst = (char*)_dst[ni].local;
	long Nsimd = _dst[ni].Nsimd;
	long simd_word = _dst[ni].simd_word;
	long word = _dst[ni].word;
	long N_ni = word / simd_word;
	long si_stride = Nsimd * simd_word;
	long o_stride = Nsimd * word;
	
	for (long si = 0; si < N_ni; si++) {
	  memcpy(&dst[si_stride*si + simd_word*_idx + o_stride*_odx],&src[_word_total*m.dst[idx] + (si_base + si)*simd_word],simd_word);
	}

	si_base += N_ni;
      }
    });
}

void cgpt_distribute::get_send_tasks_for_rank(int i, const std::map<int, std::vector<long> >& wishlists, std::vector<long>& tasks) {
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

void cgpt_distribute::send_tasks_to_ranks(const std::map<int, std::vector<long> >& wishlists, std::vector<long>& tasks) {
#ifdef USE_MPI  
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

void cgpt_distribute::wishlists_to_root(const std::vector<long>& wishlist, std::map<int, std::vector<long> >& wishlists) {
#ifdef USE_MPI
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

void cgpt_distribute::packet_prepare_need(std::vector<long>& data, const std::map<int,mp>& cr) {
  for (auto & f : cr) {
    if (f.first != rank) {
      data.push_back(f.first); // rank
      data.push_back(f.second.src.size());
    }
  }
}
