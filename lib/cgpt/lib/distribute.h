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
class global_memory_view {
 public:

  struct block_t {
    rank_t rank;
    index_t index;
    offset_t start, size;
  };

  std::vector<block_t> blocks;

  global_memory_view<offset_t,rank_t,index_t> merged() const;
};

class comm_message {
 public:
  std::vector<char> data;

  size_t offset;

  comm_message() : offset(0) {
  }

  bool eom() {
    return offset == data.size();
  }

  void get_mem(void* dst, size_t sz) {
    ASSERT(offset + sz <= data.size());
    memcpy(dst,&data[offset],sz);
    offset += sz;
  }

  size_t get_size() {
    size_t sz;
    get_mem(&sz,sizeof(sz));
    return sz;
  }

  void put_bytes(const void* v, size_t sz) {
    offset = data.size();
    data.resize(offset + sz + sizeof(sz));
    *(size_t*)&data[offset] = sz;
    memcpy(&data[offset + sizeof(sz)],v,sz);
  }

  void get_bytes(void* v, size_t sz) {
    ASSERT(get_size() == sz);
    get_mem(v,sz);
  }

  void get_bytes(std::vector<char>& buf) {
    buf.resize(get_size());
    get_mem(&buf[0],buf.size());
  }

  void put(const int v) {
    put_bytes(&v,sizeof(v));
  }

  void get(int & v) {
    get_bytes(&v,sizeof(v));
  }

  void put(const size_t v) {
    put_bytes(&v,sizeof(v));
  }

  void get(size_t& v) {
    get_bytes(&v,sizeof(v));
  }

  void put(const uint32_t v) {
    put_bytes(&v,sizeof(v));
  }

  void get(uint32_t& v) {
    get_bytes(&v,sizeof(v));
  }

  template<typename T>
  void put(const T& t) {
    std::vector<char> buf;
    convert_to_bytes(buf,t);
    put_bytes(&buf[0],buf.size());
  }

  template<typename T>
  void get(T& t) {
    std::vector<char> buf;
    get_bytes(buf);
    convert_from_bytes(t,buf);
  }

  template<typename A, typename B>
  void put(const std::map<A,B>& m) {
    put((size_t)m.size());
    for (auto & x : m)
      put(x);
  }

  template<typename A, typename B>
  void get(std::map<A,B>& m) {
    size_t sz;
    get(sz);
    for (size_t i=0;i<sz;i++) {
      std::pair<A,B> p;
      get(p);
      m.insert(p);
    }
  }

  template<typename A>
  void put(const std::vector<A>& m) {
    put((size_t)m.size());
    for (auto & x : m)
      put(x);
  }

  template<typename A>
  void get(std::vector<A>& m) {
    size_t sz;
    get(sz);
    m.resize(sz);
    for (auto & x : m)
      get(x);
  }

  template<typename A, typename B>
  void put(const std::pair<A,B>& p) {
    put(p.first);
    put(p.second);
  }

  template<typename A, typename B>
  void get(std::pair<A,B>& p) {
    get(p.first);
    get(p.second);
  }
};

template<typename rank_t>
class global_transfer {
 public:
  global_transfer(rank_t rank, Grid_MPI_Comm comm);

  rank_t rank;
  Grid_MPI_Comm comm;

  rank_t mpi_ranks, mpi_rank;
  std::vector<rank_t> mpi_rank_map;

  std::vector<MPI_Request> requests;

  template<typename data_t>
  void all_to_root(const std::vector<data_t>& my, std::map<rank_t, std::vector<data_t> > & rank);

  template<typename data_t>
  void root_to_all(const std::map<rank_t, std::vector<data_t> > & rank, std::vector<data_t>& my);

  void provide_my_receivers_get_my_senders(const std::map<rank_t, size_t>& receivers,
					   std::map<rank_t, size_t>& senders);

  void multi_send_recv(const std::map<rank_t, comm_message>& send,
		       std::map<rank_t, comm_message>& recv);

  template<typename data_t>
  void isend(rank_t other_rank, const std::vector<data_t>& data) {
#ifdef CGPT_USE_MPI
    MPI_Request r;
    ASSERT(MPI_SUCCESS == MPI_Isend(&data[0],data.size()*sizeof(data_t),MPI_CHAR,mpi_rank_map[other_rank],0x3,comm,&r));
    requests.push_back(r);
#endif
  }

  template<typename data_t>
  void irecv(rank_t other_rank, std::vector<data_t>& data) {
#ifdef CGPT_USE_MPI
    MPI_Request r;
    ASSERT(MPI_SUCCESS == MPI_Irecv(&data[0],data.size()*sizeof(data_t),MPI_CHAR,mpi_rank_map[other_rank],0x3,comm,&r));
    requests.push_back(r);
#endif
  }

  void wait();

};

template<typename offset_t, typename rank_t, typename index_t>
class global_memory_transfer : public global_transfer<rank_t> {
 public:

  typedef global_memory_view<offset_t,rank_t,index_t> view_t;

  struct block_t {
    offset_t start_dst, start_src, size;
  };

  enum memory_type {
    mt_none, mt_host, mt_accelerator
  };

  friend void convert_to_bytes(std::vector<char>& bytes, const block_t& b) {
    bytes.resize(sizeof(b));
    memcpy(&bytes[0],&b,sizeof(b));
  }

  friend void convert_from_bytes(block_t& b, const std::vector<char>& bytes) {
    ASSERT(bytes.size() == sizeof(b));
    memcpy(&b,&bytes[0],bytes.size());
  }

  // public interface
  global_memory_transfer(rank_t rank, Grid_MPI_Comm comm);

  std::map< std::pair<rank_t,rank_t>, std::map< std::pair<index_t,index_t>, std::vector<block_t> > > blocks;

  void create(const view_t& dst, const view_t& src, memory_type use_comm_buffers_of_type = mt_none);

  void execute(std::vector<std::pair<memory_type,void*>>& base_dst, 
	       std::vector<std::pair<memory_type,void*>>& base_src);

  // helper
  void print();
  void fill_blocks_from_view_pair(const view_t& dst, const view_t& src);
  void gather_my_blocks();
  void optimize();
  void create_comm_buffers(memory_type mt);
};

typedef global_memory_view<uint64_t,int,uint32_t> gm_view;
typedef global_memory_transfer<uint64_t,int,uint32_t> gm_transfer;



class cgpt_distribute {
 public:

  static const long SIMD_BASE = 256;
  static long offset(long odx, long idx) { ASSERT(idx<SIMD_BASE); return odx*SIMD_BASE + idx; };
  struct coor { int rank; long offset; };
  struct mp { std::vector<long> src; std::vector<long> dst; };
  struct plan { std::map<int,mp> cr; std::vector<long> tasks; };
  struct data_simd { void* local; long word, Nsimd, simd_word; std::vector<long> offset_data; std::vector<long> offset_buffer; };

  int rank;

  // word == sizeof(sobj), simd_word == sizeof(Coeff_t)
  cgpt_distribute(int rank, Grid_MPI_Comm comm);

  void create_plan(const std::vector<coor>& c, plan& plan) const;

  void copy_to(const plan& p, std::vector<data_simd> & src, void* dst);

  void copy_from(const plan& p, void* src, long src_size, std::vector<data_simd> & dst);

  void copy(const plan& p_dst, const plan& p_src, std::vector<data_simd> & dst, std::vector<data_simd> & src);

 protected:
  void split(const std::vector<coor>& c, std::map<int,mp>& s) const;
  Grid_MPI_Comm comm;

  int mpi_ranks, mpi_rank;
  std::vector<int> mpi_rank_map;

  // plan
  void packet_prepare_need(std::vector<long>& data, const std::map<int,mp>& cr) const;
  void wishlists_to_root(const std::vector<long>& wishlist, std::map<int, std::vector<long> >& wishlists) const;
  void send_tasks_to_ranks(const std::map<int, std::vector<long> >& wishlists, std::vector<long>& tasks) const;
  void get_send_tasks_for_rank(int i, const std::map<int, std::vector<long> >& wishlists, std::vector<long>& tasks) const;

  // copy
  void copy_data(const mp& m, std::vector<data_simd>& src, void* dst);
  void copy_data_rev(const mp& m, void* src, long src_size, std::vector<data_simd>& dst);
  void copy_remote(const std::vector<long>& tasks, const std::map<int,mp>& cr, std::vector<data_simd>& src, void* dst);
  void copy_remote_rev(const std::vector<long>& tasks, const std::map<int,mp>& cr, void* src, long src_size, std::vector<data_simd>& dst);

  // util
  long word_total(std::vector<data_simd>& src);
};
