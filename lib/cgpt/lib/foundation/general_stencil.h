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


    General stencil: arbitrary (not only axis-aligned) shifts, with the
    setup of Grid's CartesianStencil -- a lookup table per (point, outer
    site) that refers either to the local field (offset and SIMD permute) or
    to a communication buffer filled by a halo exchange.

    Split into
      - cgpt_general_stencil_geometry: type independent (grid and points);
        the lookup table and the halo transfer plan.  Shared by all fields
        (of any type) read at the same set of points.
      - cgpt_general_stencil_halo<vobj>: the buffers of one field and the
        halo exchange.

    Halo exchange in scalar sites (SIMD lanes).  A comm buffer entry is a
    full vobj whose lanes may come from different ranks (or from this rank,
    if a SIMD-split dimension is also a communication dimension), so the
    exchange is
      (1) gather:   extract the lanes to send into the transfer buffer, and
                    the lanes of this rank used in comm buffer entries into
                    its "self" section;
      (2) communicate the send sections to their ranks;
      (3) merge:    insert the lanes into the comm buffer entries.
    The transfer buffer layout is [send sections][self][recv sections].
    The communication uses Grid's StencilSendToRecvFrom interface (see
    cgpt_general_stencil_geometry::prepare and cgpt_general_stencil_exchange).

    Each rank computes both what it receives and what it sends without any
    communication: rank B needs from rank A the scalar sites
      { x + p : x in B, p a point, x + p in A }
    which rank A enumerates as { y in A : y - p in B for a point p }.  Both
    sides order this set by global lexicographic index.  Every remote site is
    transferred once per rank pair, even if several points or comm buffer
    entries use it.

    Full grids only (checkerboarded grids use the padded stencils).
*/

struct cgpt_general_stencil_entry {
  uint32_t _offset;   // outer site of the field (local) or comm buffer entry
  uint16_t _is_local;
  uint16_t _permute;  // local: bit t permutes with permute type t
};

class cgpt_general_stencil_geometry {
public:
  GridBase* grid;
  int npoints, nsimd;
  uint64_t osites;

  // lookup table, entry of point p at outer site ss is [ss * npoints + p]
  HostDeviceVector<cgpt_general_stencil_entry> entries;

  // number of comm buffer entries (vobj)
  uint64_t n_buffer;

  // halo transfer plan in scalar sites (ss * nsimd + lane); the transfer
  // buffer is [send (n_send)][self (n_self)][recv (n_recv)]
  uint64_t n_send, n_self, n_recv;
  HostDeviceVector<uint64_t> gather; // n_send + n_self: source scalar site
  HostDeviceVector<uint64_t> merge;  // n_buffer * nsimd: index relative to the self section
  std::vector<int> send_rank, recv_rank;
  std::vector<uint64_t> send_offset, send_count, recv_offset, recv_count;

  cgpt_general_stencil_geometry(GridBase* _grid, const std::vector<Coordinate>& points) :
    grid(_grid) {

    ASSERT(!grid->_isCheckerBoarded);

    int nd = grid->Nd();
    int me = grid->ThisRank();
    npoints = (int)points.size();
    nsimd = grid->Nsimd();
    osites = grid->oSites();

    ASSERT(osites < ((uint64_t)1 << 32));
    ASSERT(grid->_Nprocessors < (1 << 23));
    ASSERT(nsimd < (1 << 8));

    const Coordinate& fdims = grid->_fdimensions;

    // source of every lane: rank (24 bits), outer site (32), lane (8)
    auto key = [](uint64_t rank, uint64_t ss, uint64_t lane) {
      return (rank << 40) | (ss << 8) | lane;
    };
    auto key_rank = [](uint64_t k) { return (int)(k >> 40); };
    auto key_site = [](uint64_t k) { return (k >> 8) & 0xFFFFFFFF; };
    auto key_lane = [](uint64_t k) { return (int)(k & 0xFF); };

    // Grid's rank lookups call MPI, which must not happen in threads
    // (MPI_THREAD_SERIALIZED), so tabulate rank <-> processor coordinate
    int nranks = grid->_Nprocessors;
    std::vector<int> rank_of_processor(nranks); // lexicographic processor index
    std::vector<Coordinate> processor_of_rank(nranks);
    for (int i=0;i<nranks;i++) {
      Coordinate pcoor;
      Lexicographic::CoorFromIndex(pcoor, i, grid->_processors);
      int rank = grid->RankFromProcessorCoor(pcoor);
      rank_of_processor[i] = rank;
      processor_of_rank[rank] = pcoor;
    }

    // global coordinate of (rank, outer site, lane)
    auto global_coor = [&](int rank, int ss, int lane, Coordinate& gcoor) {
      Coordinate icoor, ocoor;
      grid->iCoorFromIindex(icoor, lane);
      grid->oCoorFromOindex(ocoor, ss);
      gcoor.resize(nd);
      for (int d=0;d<nd;d++)
	gcoor[d] = grid->_ldimensions[d] * processor_of_rank[rank][d] + grid->_rdimensions[d] * icoor[d] + ocoor[d];
    };

    // rank, outer site and lane of the global site gcoor + sign * point p
    auto neighbor = [&](const Coordinate& gcoor, int p, int sign, int& rank, int& o_idx, int& i_idx) {
      Coordinate ncoor(nd), pcoor, lcoor;
      for (int d=0;d<nd;d++)
	ncoor[d] = ((gcoor[d] + sign * points[p][d]) % fdims[d] + fdims[d]) % fdims[d];
      grid->GlobalCoorToProcessorCoorLocalCoor(pcoor, lcoor, ncoor);
      int pidx;
      Lexicographic::IndexFromCoor(pcoor, pidx, grid->_processors);
      rank = rank_of_processor[pidx];
      o_idx = grid->oIndex(lcoor);
      i_idx = grid->iIndex(lcoor);
    };

    std::vector<uint64_t> source((size_t)osites * npoints * nsimd);
    thread_for(ss, osites, {
	Coordinate gcoor;
	for (int l=0;l<nsimd;l++) {
	  global_coor(me, (int)ss, l, gcoor);
	  for (int p=0;p<npoints;p++) {
	    int rank, o_idx, i_idx;
	    neighbor(gcoor, p, +1, rank, o_idx, i_idx);
	    source[(ss * npoints + p) * nsimd + l] = key(rank, o_idx, i_idx);
	  }
	}
      });

    // lookup table; comm buffer entries with identical lane sources are shared
    entries.resize(osites * npoints);
    std::map<std::vector<uint64_t>, uint32_t> buffer_index;
    std::vector<uint64_t> buffer_source; // n_buffer * nsimd
    for (uint64_t ss=0;ss<osites;ss++) {
      for (int p=0;p<npoints;p++) {
	const uint64_t* s = &source[(ss * npoints + p) * nsimd];
	cgpt_general_stencil_entry& SE = entries[ss * npoints + p];

	// local: all lanes on this rank, in one outer site, lane l reads lane
	// l ^ x (a permute)
	bool local = true;
	int x = key_lane(s[0]);
	for (int l=0;l<nsimd && local;l++)
	  local = key_rank(s[l]) == me && key_site(s[l]) == key_site(s[0]) && (key_lane(s[l]) ^ l) == x;

	if (local) {
	  SE._offset = (uint32_t)key_site(s[0]);
	  SE._is_local = 1;
	  SE._permute = 0;
	  for (int t=0;(nsimd >> (t + 1)) > 0;t++)
	    if (x & (nsimd >> (t + 1)))
	      SE._permute |= (1 << t);
	} else {
	  std::vector<uint64_t> k(s, s + nsimd);
	  auto it = buffer_index.find(k);
	  uint32_t e;
	  if (it == buffer_index.end()) {
	    ASSERT(buffer_index.size() < ((uint64_t)1 << 32));
	    e = (uint32_t)buffer_index.size();
	    buffer_index[k] = e;
	    buffer_source.insert(buffer_source.end(), s, s + nsimd);
	  } else {
	    e = it->second;
	  }
	  SE._offset = e;
	  SE._is_local = 0;
	  SE._permute = 0;
	}
      }
    }
    n_buffer = buffer_index.size();
    buffer_index.clear();
    std::vector<uint64_t>().swap(source);

    auto global_index = [&](int rank, int ss, int lane) {
      Coordinate gcoor;
      int64_t gidx;
      global_coor(rank, ss, lane, gcoor);
      grid->GlobalCoorToGlobalIndex(gcoor, gidx);
      return gidx;
    };

    // receive: the distinct remote sites per rank, ordered by global index;
    // self: the distinct sites of this rank used in comm buffer entries
    std::map<int, std::vector<int64_t> > recv_sites;
    std::vector<uint64_t> self_sites;
    for (auto k : buffer_source) {
      int rank = key_rank(k);
      if (rank == me)
	self_sites.push_back(key_site(k) * nsimd + key_lane(k));
      else
	recv_sites[rank].push_back(global_index(rank, key_site(k), key_lane(k)));
    }

    auto sort_unique = [](auto& v) {
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
    };

    sort_unique(self_sites);
    n_self = self_sites.size();
    n_recv = 0;
    std::map<int, size_t> recv_index;
    for (auto& rs : recv_sites) {
      sort_unique(rs.second);
      recv_index[rs.first] = recv_rank.size();
      recv_rank.push_back(rs.first);
      recv_offset.push_back(n_self + n_recv);
      recv_count.push_back(rs.second.size());
      n_recv += rs.second.size();
    }

    merge.resize(n_buffer * nsimd);
    for (uint64_t i=0;i<n_buffer * nsimd;i++) {
      uint64_t k = buffer_source[i];
      int rank = key_rank(k);
      if (rank == me) {
	uint64_t s = key_site(k) * nsimd + key_lane(k);
	merge[i] = std::lower_bound(self_sites.begin(), self_sites.end(), s) - self_sites.begin();
      } else {
	auto& rs = recv_sites[rank];
	int64_t gidx = global_index(rank, key_site(k), key_lane(k));
	size_t r = recv_index[rank];
	merge[i] = recv_offset[r] + (std::lower_bound(rs.begin(), rs.end(), gidx) - rs.begin());
      }
    }

    // send: the sites y of this rank with y - p on another rank for a point
    // p, per rank ordered by global index
    std::map<int, std::vector<std::pair<int64_t, uint64_t> > > send_sites;
    if (grid->_Nprocessors > 1) {
      Coordinate gcoor;
      for (uint64_t ss=0;ss<osites;ss++) {
	for (int l=0;l<nsimd;l++) {
	  global_coor(me, (int)ss, l, gcoor);
	  int64_t gidx;
	  grid->GlobalCoorToGlobalIndex(gcoor, gidx);
	  for (int p=0;p<npoints;p++) {
	    int rank, o_idx, i_idx;
	    neighbor(gcoor, p, -1, rank, o_idx, i_idx);
	    if (rank != me)
	      send_sites[rank].push_back({gidx, ss * nsimd + l});
	  }
	}
      }
    }

    n_send = 0;
    for (auto& ss : send_sites) {
      sort_unique(ss.second);
      send_rank.push_back(ss.first);
      send_offset.push_back(n_send);
      send_count.push_back(ss.second.size());
      n_send += ss.second.size();
    }

    gather.resize(n_send + n_self);
    uint64_t i = 0;
    for (auto& ss : send_sites)
      for (auto& s : ss.second)
	gather[i++] = s.second;
    for (auto s : self_sites)
      gather[i++] = s;

    // the transfer buffer has the same size on all ranks and peers learn
    // where to put (or get) their data (see communication below)
    n_transfer = n_send + n_self + n_recv;
    RealD max_transfer = (RealD)n_transfer;
    RealD max_peers = (RealD)(send_rank.size() + recv_rank.size());
    grid->GlobalMax(max_transfer);
    grid->GlobalMax(max_peers);
    n_transfer = (uint64_t)max_transfer;
    communicates = max_peers > 0;
    exchange_offsets();

    if (getenv("CGPT_GENERAL_STENCIL_DEBUG"))
      std::cout << GridLogMessage << "General stencil with " << npoints << " points: "
		<< n_buffer << " comm buffer entries, " << n_send << " sites sent to "
		<< send_rank.size() << " ranks, " << n_recv << " received from "
		<< recv_rank.size() << " ranks, " << n_self << " from self" << std::endl;

    entries.toDevice();
    if (gather.size())
      gather.toDevice();
    if (merge.size())
      merge.toDevice();
  }

  bool has_halo() const {
    return n_buffer > 0;
  }

  // Communication through the StencilSendToRecvFrom interface of Grid's
  // communicator: MPI between nodes (or always with --shm-mpi 1, Grid's
  // default), staged through host memory without accelerator-aware MPI, and
  // puts (gets with NVLINK_GET) into the peer's transfer buffer within a
  // node.  A put addresses the peer's buffer at the same offset of its
  // shared-memory window as the local address passed, so
  //   - the transfer buffers live in Grid's shared-memory heap, are allocated
  //     in the same order and with the same size (n_transfer) on all ranks
  //     (as Grid's stencils, they are only used during an exchange, so a later
  //     reset of the heap by another stencil does no harm), and
  //   - each rank learns at setup where its peers receive its data
  //     (send_remote_offset) and where they hold the data they send to it
  //     (recv_remote_offset), in scalar sites from the transfer buffer start.
  bool communicates; // any rank has peers (the same on all ranks)
  uint64_t n_transfer;
  std::vector<uint64_t> send_remote_offset, recv_remote_offset;

  void exchange_offsets() {
    send_remote_offset.resize(send_rank.size());
    recv_remote_offset.resize(recv_rank.size());
    if (!communicates)
      return;
#ifdef CGPT_USE_MPI
    // tag_recv: a receiver tells a sender where it receives the data,
    // tag_send: a sender tells a receiver where it holds the data
    const int tag_recv = 0x6573, tag_send = 0x6574;
    std::vector<uint64_t> my_recv_offset(recv_rank.size());
    std::vector<MPI_Request> requests;
    MPI_Request rq;
    for (size_t r=0;r<recv_rank.size();r++) {
      my_recv_offset[r] = n_send + recv_offset[r];
      ASSERT(MPI_SUCCESS == MPI_Isend(&my_recv_offset[r], 1, MPI_UINT64_T, recv_rank[r], tag_recv, grid->communicator, &rq));
      requests.push_back(rq);
      ASSERT(MPI_SUCCESS == MPI_Irecv(&recv_remote_offset[r], 1, MPI_UINT64_T, recv_rank[r], tag_send, grid->communicator, &rq));
      requests.push_back(rq);
    }
    for (size_t r=0;r<send_rank.size();r++) {
      ASSERT(MPI_SUCCESS == MPI_Isend(&send_offset[r], 1, MPI_UINT64_T, send_rank[r], tag_send, grid->communicator, &rq));
      requests.push_back(rq);
      ASSERT(MPI_SUCCESS == MPI_Irecv(&send_remote_offset[r], 1, MPI_UINT64_T, send_rank[r], tag_recv, grid->communicator, &rq));
      requests.push_back(rq);
    }
    ASSERT(MPI_SUCCESS == MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE));
#else
    ERR("General stencil needs communication but cgpt was compiled without MPI");
#endif
  }

  // the packets of one exchange (transfer: this rank's transfer buffer):
  // f(xmit, recv, rank, do_xmit, do_recv, xbytes, rbytes), a send packet
  // with recv at the peer's offset and a receive packet with xmit at the
  // peer's offset (for puts and gets, respectively)
  template<typename F>
  void packets(char* transfer, size_t site_bytes, F f) {
    for (size_t r=0;r<send_rank.size();r++)
      f(transfer + send_offset[r] * site_bytes, transfer + send_remote_offset[r] * site_bytes,
	send_rank[r], 1, 0, send_count[r] * site_bytes, (uint64_t)0);
    for (size_t r=0;r<recv_rank.size();r++)
      f(transfer + recv_remote_offset[r] * site_bytes, transfer + (n_send + recv_offset[r]) * site_bytes,
	recv_rank[r], 0, 1, (uint64_t)0, recv_count[r] * site_bytes);
  }

  // (dir: distinguishes the exchanges of one batch, < 32)
  void prepare(std::vector<CommsRequest_t>& list, char* transfer, size_t site_bytes, int dir) {
    packets(transfer, site_bytes, [&](char* xmit, char* recv, int rank, int dox, int dor, uint64_t xbytes, uint64_t rbytes) {
	grid->StencilSendToRecvFromPrepare(list, xmit, rank, dox, recv, rank, dor, xbytes, rbytes, dir);
      });
  }

  void begin(std::vector<CommsRequest_t>& list, char* transfer, size_t site_bytes, int dir) {
    packets(transfer, site_bytes, [&](char* xmit, char* recv, int rank, int dox, int dor, uint64_t xbytes, uint64_t rbytes) {
	grid->StencilSendToRecvFromBegin(list, xmit, xmit, rank, dox, recv, recv, rank, dor, xbytes, rbytes, dir);
      });
  }
};

// geometries shared by fields with the same point set (also across types)
class cgpt_general_stencil_geometry_cache {
public:
  GridBase* grid;
  std::map<std::vector<std::vector<int> >, std::shared_ptr<cgpt_general_stencil_geometry> > geometries;

  cgpt_general_stencil_geometry_cache(GridBase* _grid) : grid(_grid) {
  }

  std::shared_ptr<cgpt_general_stencil_geometry> get(const std::vector<Coordinate>& points) {
    std::vector<std::vector<int> > k;
    for (auto& p : points)
      k.push_back(std::vector<int>(p.begin(), p.end()));
    auto it = geometries.find(k);
    if (it != geometries.end())
      return it->second;
    auto g = std::make_shared<cgpt_general_stencil_geometry>(grid, points);
    geometries[k] = g;
    return g;
  }
};

// what a kernel needs per field
template<typename vobj>
struct cgpt_general_stencil_view {
  const cgpt_general_stencil_entry* entries;
  const vobj* buffer;
  int npoints;
};

// The halo exchange of one field: gather, communicate (the packets of many
// fields are communicated together, see cgpt_general_stencil_exchange),
// merge.
template<typename vobj>
class cgpt_general_stencil_halo {
public:
  typedef typename vobj::scalar_object sobj;

  std::shared_ptr<cgpt_general_stencil_geometry> geometry;
  vobj* buffer;
  sobj* transfer; // in Grid's shared-memory heap if the geometry communicates
  size_t buffer_bytes, transfer_bytes;

  cgpt_general_stencil_halo(std::shared_ptr<cgpt_general_stencil_geometry> _geometry) :
    geometry(_geometry) {
    auto& g = *geometry;
    ASSERT(vobj::Nsimd() == g.nsimd);
    buffer_bytes = g.n_buffer * sizeof(vobj);
    buffer = buffer_bytes ? (vobj*)MemoryManager::AcceleratorAllocate(buffer_bytes) : 0;
    if (g.communicates) {
      // same size on all ranks (see geometry)
      transfer_bytes = 0;
      transfer = (sobj*)g.grid->ShmBufferMalloc(g.n_transfer * sizeof(sobj));
    } else {
      transfer_bytes = g.n_self * sizeof(sobj);
      transfer = transfer_bytes ? (sobj*)MemoryManager::AcceleratorAllocate(transfer_bytes) : 0;
    }
  }

  ~cgpt_general_stencil_halo() {
    if (buffer)
      MemoryManager::AcceleratorFree(buffer, buffer_bytes);
    if (transfer_bytes)
      MemoryManager::AcceleratorFree(transfer, transfer_bytes);
  }

  void gather(const Lattice<vobj>& field) {
    auto& g = *geometry;
    ASSERT(field.Grid() == g.grid);
    if (g.n_send + g.n_self == 0)
      return;
    int nsimd = g.nsimd;
    sobj* p_transfer = transfer;
    autoView(field_v, field, AcceleratorRead);
    const uint64_t* p_gather = g.gather.device;
    accelerator_for(i, g.n_send + g.n_self, 1, {
	uint64_t s = p_gather[i];
	p_transfer[i] = extractLane(s % nsimd, field_v[s / nsimd]);
      });
  }

  void merge() {
    auto& g = *geometry;
    if (!g.has_halo())
      return;
    int nsimd = g.nsimd;
    sobj* p_transfer = transfer;
    vobj* p_buffer = buffer;
    const uint64_t* p_merge = g.merge.device;
    uint64_t self = g.n_send;
    accelerator_for(i, g.n_buffer * nsimd, 1, {
	insertLane(i % nsimd, p_buffer[i / nsimd], p_transfer[self + p_merge[i]]);
      });
  }

  cgpt_general_stencil_view<vobj> view() const {
    cgpt_general_stencil_view<vobj> v;
    v.entries = geometry->entries.device;
    v.buffer = buffer;
    v.npoints = geometry->npoints;
    return v;
  }
};

// Halo exchange of several fields with one communication phase, in the
// sequence of Grid's CartesianStencil (HaloGather, CommunicateBegin,
// CommunicateComplete):
//
//   gather all, barrier, prepare, poll device-to-host, begin, poll receives,
//   complete, (barrier), merge all
//
// The barrier after the gather makes sure that no rank still uses the
// memory a put (get) writes (reads), e.g. in the merge of an earlier
// exchange in the same heap region.  After the transfers, every rank needs
// one barrier, which StencilSendToRecvFromComplete does itself (without
// NVLINK_GET) except with accelerator-aware MPI and no MPI requests on this
// rank; then (and with NVLINK_GET, where a sender must not overwrite its
// buffer before the gets are done) it is done here.
//
// The calls must be made by all ranks of the grid in the same order
// (barriers): `communicates` is the same on all ranks.
template<typename halo_t, typename field_t>
void cgpt_general_stencil_exchange(const std::vector<std::pair<halo_t*, const field_t*> >& halos) {

  typedef typename halo_t::sobj sobj;
  const size_t max_dir = 32; // tags of Grid's communicator: dir + rank * 32

  // without communication (the same halos on all ranks have it)
  std::vector<std::pair<halo_t*, const field_t*> > comm;
  for (auto& h : halos) {
    if (h.first->geometry->communicates) {
      comm.push_back(h);
    } else {
      h.first->gather(*h.second);
      h.first->merge();
    }
  }

  for (size_t i0=0;i0<comm.size();i0+=max_dir) {
    size_t i1 = std::min(comm.size(), i0 + max_dir);
    GridBase* grid = comm[i0].first->geometry->grid;

    for (size_t i=i0;i<i1;i++)
      comm[i].first->gather(*comm[i].second);

    std::vector<CommsRequest_t> list;
    grid->StencilBarrier();
    for (size_t i=i0;i<i1;i++)
      comm[i].first->geometry->prepare(list, (char*)comm[i].first->transfer, sizeof(sobj), (int)(i - i0));
    grid->StencilSendToRecvFromPollDtoH(list);
    acceleratorCopySynchronise();
    for (size_t i=i0;i<i1;i++)
      comm[i].first->geometry->begin(list, (char*)comm[i].first->transfer, sizeof(sobj), (int)(i - i0));
    grid->StencilSendToRecvFromPollIRecv(list);
#if defined(ACCELERATOR_AWARE_MPI)
    bool barrier = list.size() == 0;
#elif defined(NVLINK_GET)
    bool barrier = true;
#else
    bool barrier = false;
#endif
    grid->StencilSendToRecvFromComplete(list, 0);
    if (barrier)
      grid->StencilBarrier();

    for (size_t i=i0;i<i1;i++)
      comm[i].first->merge();
  }
}

// The general stencils of a kernel: each field has its own set of
// (non-zero) points; fields with the same set share a geometry.
template<typename vobj>
class cgpt_general_stencil_manager {
public:
  GridBase* grid;
  std::vector<Coordinate> shifts;
  std::map<int, std::set<int> > field_points;
  std::map<int, std::map<int, int> > field_point_map;
  std::vector<std::shared_ptr<cgpt_general_stencil_halo<vobj> > > halos; // per field index
  HostDeviceVector<cgpt_general_stencil_view<vobj> > views; // per field index (n_fields)
  std::shared_ptr<cgpt_general_stencil_geometry_cache> cache;

  cgpt_general_stencil_manager(GridBase* _grid, const std::vector<Coordinate>& _shifts,
			       std::shared_ptr<cgpt_general_stencil_geometry_cache> _cache = 0) :
    grid(_grid), shifts(_shifts), cache(_cache) {
    if (!cache)
      cache = std::make_shared<cgpt_general_stencil_geometry_cache>(grid);
  }

  bool is_trivial(int point) {
    for (auto s : shifts[point])
      if (s != 0)
	return false;
    return true;
  }

  void register_point(int index, int point) {
    if (is_trivial(point))
      return;
    field_points[index].insert(point);
  }

  // n_fields: number of fields of the kernel (views); reset_shm: start a new
  // allocation sequence in Grid's shared-memory heap (as the first stencil of
  // a CartesianStencilManager), a caller with several managers (e.g. matrix
  // and vector fields) resets only in the first one
  void create_stencils(int n_fields, bool reset_shm = true) {
    std::map<int, std::shared_ptr<cgpt_general_stencil_geometry> > geometries;
    bool communicates = false;
    for (auto& fp : field_points) {
      int index = fp.first;
      std::vector<Coordinate> points;
      for (auto p : fp.second) {
	field_point_map[index][p] = (int)points.size();
	points.push_back(shifts[p]);
      }
      geometries[index] = cache->get(points);
      communicates = communicates || geometries[index]->communicates;
    }

    if (communicates && reset_shm)
      grid->ShmBufferFreeAll();

    for (auto& ig : geometries) {
      int index = ig.first;
      if ((int)halos.size() < index + 1)
	halos.resize(index + 1);
      halos[index] = std::make_shared<cgpt_general_stencil_halo<vobj> >(ig.second);
    }

    // unused fields: no entries
    views.resize(n_fields);
    for (int i=0;i<n_fields;i++) {
      if (i < (int)halos.size() && halos[i]) {
	views[i] = halos[i]->view();
      } else {
	views[i].entries = 0;
	views[i].buffer = 0;
	views[i].npoints = 0;
      }
    }
    views.toDevice();
  }

  int map_point(int index, int point) {
    if (is_trivial(point))
      return -1;
    return field_point_map[index][point];
  }

  // the exchanges this manager contributes (fields: indexed as the halos)
  void exchanges(std::vector<std::pair<cgpt_general_stencil_halo<vobj>*, const Lattice<vobj>*> >& ex,
		 const PVector<Lattice<vobj> >& fields) {
    for (int i=0;i<(int)halos.size();i++)
      if (halos[i])
	ex.push_back({halos[i].get(), &fields[i]});
  }

  void exchange(const PVector<Lattice<vobj> >& fields) {
    std::vector<std::pair<cgpt_general_stencil_halo<vobj>*, const Lattice<vobj>*> > ex;
    exchanges(ex, fields);
    cgpt_general_stencil_exchange(ex);
  }
};
