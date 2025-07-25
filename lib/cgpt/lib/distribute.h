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

enum memory_type {
  mt_host = 0x0,
  mt_accelerator = 0x1,
  mt_none = 0x2
};

#define mt_int_len mt_none

static memory_type cgpt_memory_type_from_string(const std::string& s) {
  if (s == "none") {
    return mt_none;
  } else if (s == "host") {
    return mt_host;
  } else if (s == "accelerator") {
    return mt_accelerator;
  } else {
    ERR("Unknown memory_type %s",s.c_str());
  }
}

template<typename offset_t, typename rank_t, typename index_t>
class global_memory_view {
 public:

  struct block_t {
    rank_t rank;
    index_t index;
    offset_t start;
  };

  AlignedVector<block_t> blocks;
  offset_t block_size;

  void print() const;
  offset_t size() const;
  bool is_aligned() const;

  void operator=(const global_memory_view<offset_t,rank_t,index_t>& other);
};

class comm_message {
 public:
  AlignedVector<char> data;

  size_t offset, size;

  comm_message() : offset(0), size(0) {
  }

  bool eom() {
    return offset == data.size();
  }

  void alloc() {
    data.resize(size);
  }

  void reserve(size_t sz) {
    size += sz;
  }

  void resize(size_t sz) {
    reserve(sz);
    alloc();
  }

  void reset() {
    offset = 0;
  }
  
  void* get(size_t sz) {
    void* r = (void*)&data[offset];
    size_t n = offset + sz;
    ASSERT(n <= data.size());
    offset = n;
    return r;
  }

  template<typename data_t>
  void put(data_t & data) {
    memcpy(get(sizeof(data)),&data,sizeof(data));
  }
};

template<typename rank_t>
class global_transfer {
 public:
  global_transfer(rank_t rank, Grid_MPI_Comm comm);

  ~global_transfer() {
#ifndef ACCELERATOR_AWARE_MPI
    host_bounce_cleanup();
#endif
  }

  const size_t size_mpi_max = INT_MAX;
  rank_t rank;
  Grid_MPI_Comm comm;

  rank_t mpi_ranks, mpi_rank;
  std::vector<rank_t> mpi_rank_map;

#ifdef CGPT_USE_MPI
  std::vector<MPI_Request> requests;
#endif

  template<typename vec_t>
  void all_to_root(const vec_t & my, std::map<rank_t, vec_t > & rank);

  template<typename vec_t>
  void root_to_all(const std::map<rank_t, vec_t > & rank, vec_t& my);

  template<typename vec_t>
  void global_sum(vec_t & data) {
    global_sum(&data[0], data.size());
  }
  
  void global_sum(uint64_t* pdata, size_t size);

  long global_gcd(long n);
  
  void provide_my_receivers_get_my_senders(const std::map<rank_t, size_t>& receivers,
					   std::map<rank_t, size_t>& senders);

  void multi_send_recv(const std::map<rank_t, comm_message>& send,
		       std::map<rank_t, comm_message>& recv);


  void isend(rank_t other_rank, const void* pdata, size_t sz, memory_type type);
  void irecv(rank_t other_rank, void* pdata, size_t sz, memory_type type);

  template<typename vec_t>
  void isend(rank_t other_rank, const vec_t& data) {
    isend(other_rank,&data[0],data.size()*sizeof(data[0]), mt_host);
  }

  template<typename vec_t>
  void irecv(rank_t other_rank, vec_t& data) {
    irecv(other_rank,&data[0],data.size()*sizeof(data[0]), mt_host);
  }

  void waitall();

#ifndef ACCELERATOR_AWARE_MPI
  struct hostBounceBuffer_t { void* host; void * device; size_t size; size_t reserved; memory_type device_mt; int tag; rank_t sender; };
  std::vector<hostBounceBuffer_t> host_bounce_buffer;
  std::map<uint64_t, uint64_t> host_checksum_index;

  uint64_t host_checksum_increment(rank_t sender, rank_t receiver) {
    uint64_t tg = ((uint64_t)sender << (uint64_t)32) ^ ((uint64_t)receiver);
    if (auto search = host_checksum_index.find(tg); search != host_checksum_index.end()) {
      search->second ++;
      return search->second;
    } else {
      host_checksum_index[tg] = 0;
      return 0;
    }
  }

  void host_bounce_cleanup() {
    for (auto & b : host_bounce_buffer) {
      	acceleratorFreeCpu(b.host);
    }
    host_bounce_buffer.clear();
  }
  
  void host_bounce_reset() {
    for (auto & b : host_bounce_buffer) {
      b.size = 0;
      b.device = 0;
    }
  }

  uint64_t host_bounce_checksum(uint64_t* pdata, size_t n, memory_type mt) {
    if (mt == mt_accelerator) {
      return checksum_gpu(pdata, n);
    } else {
      uint64_t v = 0;
      thread_region
	{
	  uint64_t vt = 0;
	  thread_for_in_region(i, n, {
	      auto l = i % 61;
	      vt ^= pdata[i]<<l | pdata[i]>>(64-l);
	    });
	  thread_critical
	    {
	      v ^= vt;
	    }
	}
      return v;
    }
  }

  void* host_bounce_allocate(size_t sz, void* device, memory_type device_mt, int tag, rank_t sender) {
    for (auto & b : host_bounce_buffer) {
      if (b.size == 0) {
	if (b.reserved < sz) {
	  acceleratorFreeCpu(b.host);
	  b.host = acceleratorAllocCpu(sz);
	  b.reserved = sz;
	}
	b.size = sz;
	b.device = device;
	b.device_mt = device_mt;
	b.tag = tag;
	b.sender = sender;
	return b.host;
      }
    }

    hostBounceBuffer_t bb;
    host_bounce_buffer.push_back(bb);

    auto & b = host_bounce_buffer.back();
    b.host = acceleratorAllocCpu(sz);
    b.reserved = sz;
    b.device = device;
    b.size = sz;
    b.device_mt = device_mt;
    b.tag = tag;
    b.sender = sender;
    return b.host;
  }
#endif

};

template<typename offset_t, typename rank_t, typename index_t>
class global_memory_transfer : public global_transfer<rank_t> {
 public:

  typedef global_memory_view<offset_t,rank_t,index_t> view_t;

  struct block_t {
    offset_t start_dst, start_src;
  };

  struct rank_pair_t {
    rank_t dst_rank;
    rank_t src_rank;

    bool operator<(const rank_pair_t & other) const {
      if (dst_rank < other.dst_rank)
	return true;
      else if (dst_rank > other.dst_rank)
	return false;
      if (src_rank < other.src_rank)
	return true;
      return false;
    }
  };

  struct index_pair_t {
    index_t dst_index;
    index_t src_index;

    bool operator<(const index_pair_t & other) const {
      if (dst_index < other.dst_index)
	return true;
      else if (dst_index > other.dst_index)
	return false;
      if (src_index < other.src_index)
	return true;
      return false;
    }
  };

  typedef HostDeviceVector<block_t> blocks_t;
  typedef std::vector<block_t> thread_blocks_t;

  struct abstract_blocks_t {
    size_t n_blocks;
    size_t offset;
  };

  class memory_view {
  public:
    memory_type type;
    void* ptr;
    size_t sz;
  };

  class memory_buffer {
  public:
    memory_view view;

    memory_buffer(size_t _sz, memory_type _type) {
      view.sz = _sz;
      view.type = _type;
      if (_type == mt_accelerator) {
	view.ptr = acceleratorAllocDevice(_sz);
      } else if (_type == mt_host) {
	view.ptr = acceleratorAllocCpu(_sz);
      } else {
	ERR("Unknown memory type");
      }
    }

    memory_buffer(const memory_buffer& other) {
      ERR("Copy constructor not yet supported");
    }

    memory_buffer(memory_buffer&& other) {
      // move constructor
      view = other.view;
      other.view.ptr = 0;
      other.view.type = mt_none;
      other.view.sz = 0;
    }

    ~memory_buffer() {
      if (view.type == mt_accelerator) {
	acceleratorFreeDevice(view.ptr);
      } else if (view.type == mt_host) {
	acceleratorFreeCpu(view.ptr);
      }
    }
  };

  template<typename vec_t>
  friend void convert_to_bytes(vec_t& bytes, const block_t& b) {
    ASSERT(sizeof(bytes[0])==1);
    bytes.resize(sizeof(b));
    memcpy(&bytes[0],&b,sizeof(b));
  }

  template<typename vec_t>
  friend void convert_from_bytes(block_t& b, const vec_t& bytes) {
    ASSERT(bytes.size() == sizeof(b) && sizeof(bytes[0])==1);
    memcpy(&b,&bytes[0],bytes.size());
  }

  // memory buffers
  std::vector<memory_buffer> buffers;
  std::map<rank_t, memory_view> send_buffers, recv_buffers;
  memory_type comm_buffers_type;
  std::map<rank_t, std::map< index_t, thread_blocks_t > > send_blocks, recv_blocks;
  std::map<rank_t, std::map< index_t, blocks_t > > send_blocks_hd, recv_blocks_hd;

  // bounds and alignment
  std::vector<offset_t> bounds_dst, bounds_src, alignment;
  offset_t global_alignment;

  // public interface
  global_memory_transfer(rank_t rank, Grid_MPI_Comm comm);

  offset_t block_size;
  std::map< rank_pair_t , std::map< index_pair_t, thread_blocks_t > > blocks;
  std::map< rank_pair_t , std::map< index_pair_t, blocks_t > > blocks_hd;

  template<typename A, typename B>
  void populate_hd(std::map<A, std::map<B, blocks_t>>& b_hd,
		   const std::map<A, std::map<B, thread_blocks_t>>& b) {
    for (auto & ta : b) {
      auto & da = b_hd[ta.first];
      for (auto & tb : ta.second) {
	auto & db = da[tb.first];
	db.resize(tb.second.size());
	thread_for(i, tb.second.size(), {
	    db[i] = tb.second[i];
	  });
	db.toDevice();
      }
    }
  }

  void create(const view_t& dst, const view_t& src,
	      memory_type use_comm_buffers_of_type = mt_none,
	      bool local_only = false,
	      bool skip_optimize = false);

  void execute(std::vector<memory_view>& base_dst, 
	       std::vector<memory_view>& base_src);

  // helper
  void print();
  void fill_blocks_from_view_pair(const view_t& dst, const view_t& src, bool local_only);
  void gather_my_blocks();
  void optimize();
  long optimize(thread_blocks_t& blocks);
  void skip(thread_blocks_t& blocks, long gcd);
  void create_bounds_and_alignment();
  void create_comm_buffers(memory_type mt);

  template<typename ranks_t>
  void prepare_comm_message(comm_message & msg, ranks_t & ranks, bool populate);
  void merge_comm_blocks(std::map<rank_t, comm_message> & src);

  template<typename K, typename V1, typename V2>
  void distribute_merge_into(std::map<K,V1> & target, const std::map<K,V2> & src);
  void distribute_merge_into(thread_blocks_t & target, const thread_blocks_t & src);

  struct bcopy_arg_t {
    const blocks_t& blocks;
    memory_view& base_dst; 
    const memory_view& base_src;
  };

  struct bcopy_ptr_arg_t {
    const blocks_t & blocks;
    char* p_dst;
    const char* p_src;
  };

void bcopy(const std::vector<bcopy_arg_t>& args);
};

typedef global_memory_view<uint64_t,int,uint32_t> gm_view;
typedef global_memory_transfer<uint64_t,int,uint32_t> gm_transfer;

// perform a test of the global memory system
void test_global_memory_system();
