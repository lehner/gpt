/*
    GPT - Grid Python Toolkit
    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2023  Mattia Bruno

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
  
struct cgpt_stencil_matrix_factor_t {
  int index; // index of field
  int point; // index of shift
  int adj; // adjoint of matrix
  int temporary; // field is a per-site temporary (set at creation)
  int zero_point; // shift is zero (set at creation, comm_type 1 and 2)
};

struct cgpt_stencil_matrix_code_offload_t {
  int target;
  int accumulate;
  int target_temporary;
  int accumulate_temporary;
  int unit_weight; // weight == 1: skip the multiplication
  ComplexD weight;
  int size;
  cgpt_stencil_matrix_factor_t* factor;
};

struct cgpt_stencil_matrix_code_t {
  int target; // target field index
  int accumulate; // field index to accumulate or -1
  ComplexD weight; // weight of term
  std::vector<cgpt_stencil_matrix_factor_t> factor; 
};

class cgpt_stencil_matrix_base {
 public:
  virtual ~cgpt_stencil_matrix_base() {};
  virtual void execute(const std::vector<cgpt_Lattice_base*>& fields, int fast_osites) = 0;
};

template<typename T>
class cgpt_stencil_matrix : public cgpt_stencil_matrix_base {
 public:

  typedef CartesianStencil<T, T, SimpleStencilParams> CartesianStencil_t;
  typedef CartesianStencilView<T, T, SimpleStencilParams> CartesianStencilView_t;
  
  HostDeviceVector<cgpt_stencil_matrix_code_offload_t> code;
  HostDeviceVector<cgpt_stencil_matrix_factor_t> factors;
    
  int n_code_parallel_block_size, n_code_parallel_blocks;
  int comm_type; // 0: cartesian stencil, 1: no communication, 2: general stencil

  // per-site temporaries: owned by the stencil (a device-only buffer of one
  // block of osites_per_cache_block outer sites per temporary, allocated
  // once); the kernel runs block by block and addresses temporaries relative
  // to the block start.  The caller passes only the other fields.
  bool has_temporaries;
  uint64_t osites_per_cache_block;
  T* temp_buffer;
  size_t temp_bytes;

  // comm_type == 1: local stencil (no communication)
  cgpt_GeneralLocalStencil* general_local_stencil;

  // comm_type == 2: general stencil (arbitrary points, halo exchange)
  cgpt_general_stencil_manager<T>* gsm;

  // comm_type == 0: cartesian stencil
  SimpleCompressor<T>* compressor;
  CartesianStencilManager<CartesianStencil_t>* sm;
  
  cgpt_stencil_matrix(GridBase* grid,
		      const std::vector<Coordinate>& shifts,
		      const std::vector<cgpt_stencil_matrix_code_t>& _code,
		      int _n_code_parallel_block_size,
		      int _comm_type,
		      const std::vector<int>& temporaries,
		      long _osites_per_cache_block) :
    code(_code.size()), comm_type(_comm_type),
    n_code_parallel_block_size(_n_code_parallel_block_size) {

    ASSERT(_code.size() % n_code_parallel_block_size == 0);
    n_code_parallel_blocks = (int)_code.size() / n_code_parallel_block_size;
    
    // total number of factors
    int nfactors = 0;
    for (int i=0;i<_code.size();i++)
      nfactors += (int)_code[i].factor.size();
    factors.resize(nfactors);
    // fill in code and factors and link them
    nfactors = 0;
    auto is_temporary = [&](int f) {
      return std::find(temporaries.begin(), temporaries.end(), f) != temporaries.end();
    };
    has_temporaries = temporaries.size() > 0;
    if (has_temporaries) {
      ASSERT(_comm_type != 0);
      ASSERT(n_code_parallel_blocks == 1);
    }

    // index remapping: temporaries to their buffer slot, all other fields to
    // their position among the fields the caller passes (in index order)
    int n_indices = 0;
    for (auto & c : _code) {
      n_indices = std::max(n_indices, std::max(c.target, c.accumulate) + 1);
      for (auto & f : c.factor)
	n_indices = std::max(n_indices, f.index + 1);
    }
    for (auto t : temporaries)
      n_indices = std::max(n_indices, t + 1);
    std::vector<int> remap(n_indices);
    int n_passed = 0, n_temporaries = 0;
    for (int f=0;f<n_indices;f++)
      remap[f] = is_temporary(f) ? n_temporaries++ : n_passed++;
    for (int i=0;i<_code.size();i++) {
      code[i].target = remap[_code[i].target];
      code[i].accumulate = _code[i].accumulate == -1 ? -1 : remap[_code[i].accumulate];
      code[i].target_temporary = is_temporary(_code[i].target);
      code[i].accumulate_temporary = _code[i].accumulate != -1 && is_temporary(_code[i].accumulate);
      code[i].weight = _code[i].weight;
      code[i].unit_weight = _code[i].weight == ComplexD(1.0, 0.0);
      code[i].size = (int)_code[i].factor.size();
      code[i].factor = &factors.device[nfactors];
      memcpy(&factors[nfactors], &_code[i].factor[0], sizeof(cgpt_stencil_matrix_factor_t) * code[i].size);
      for (int j=0;j<code[i].size;j++) {
	auto & f = factors[nfactors + j];
	bool zero = true;
	for (auto x : shifts[f.point])
	  zero = zero && (x == 0);
	f.zero_point = comm_type != 0 && zero;
	f.temporary = is_temporary(f.index);
	if (f.temporary)
	  ASSERT(zero); // temporaries are per site
	f.index = remap[f.index];
      }
      nfactors += code[i].size;
    }

    uint64_t osites = grid->oSites();
    if (_osites_per_cache_block > 0) {
      osites_per_cache_block = (uint64_t)_osites_per_cache_block;
    } else {
#ifdef GRID_HAS_ACCELERATOR
      // large launches for parallelism, bounded to limit the buffer size
      osites_per_cache_block = 65536;
#else
      // about 2 MB of temporaries per block
      uint64_t bytes = (uint64_t)std::max((size_t)1, temporaries.size()) * sizeof(T);
      osites_per_cache_block = std::max((uint64_t)64, (uint64_t)(2 * 1024 * 1024) / bytes);
#endif
    }
    osites_per_cache_block = std::min(osites_per_cache_block, osites);

    temp_bytes = (size_t)n_temporaries * osites_per_cache_block * sizeof(T);
    temp_buffer = temp_bytes ? (T*)MemoryManager::AcceleratorAllocate(temp_bytes) : 0;

    if (comm_type == 1) {
      general_local_stencil = new cgpt_GeneralLocalStencil(grid,shifts,-1);
    } else if (comm_type == 2) {

      gsm = new cgpt_general_stencil_manager<T>(grid, shifts);

      // (temporaries are read at the zero point, which needs no stencil)
      for (int i=0;i<nfactors;i++)
	gsm->register_point(factors[i].index, factors[i].point);

      gsm->create_stencils(n_passed);

      for (int i=0;i<nfactors;i++)
	factors[i].point = gsm->map_point(factors[i].index, factors[i].point);

    } else {

      sm = new CartesianStencilManager<CartesianStencil_t>(grid, shifts);

      // for all factors that require a non-trivial shift, create a separate stencil object
      for (int i=0;i<nfactors;i++) {
	sm->register_point(factors[i].index, factors[i].point);
      }

      sm->create_stencils(true, Even);

      for (int i=0;i<nfactors;i++) {
	factors[i].point = sm->map_point(factors[i].index, factors[i].point);
      }      

      compressor = new SimpleCompressor<T>();
    }

    factors.toDevice();
    code.toDevice();
  }

  virtual ~cgpt_stencil_matrix() {
    if (temp_buffer)
      MemoryManager::AcceleratorFree(temp_buffer, temp_bytes);
    if (comm_type == 1) {
      delete general_local_stencil;
    } else if (comm_type == 2) {
      delete gsm;
    } else {
      delete compressor;
      delete sm;
    }
  }
 
  virtual void execute(PVector<Lattice<T>> &fields, int fast_osites) {

    VECTOR_VIEW_OPEN(fields,fields_v,AcceleratorWrite);

    int n_code = code.size();
    const cgpt_stencil_matrix_code_offload_t* p_code = code.device;

    typedef decltype(coalescedRead(fields_v[0][0])) obj_t;

    int nd = fields[0].Grid()->Nd();

    int _npb = n_code_parallel_blocks;
    int _npbs = n_code_parallel_block_size;

    uint64_t osites = fields[0].Grid()->oSites();

    int _fast_osites = fast_osites;
    
    if (comm_type != 0) {

      // factor fetch: temporaries (relative to the cache block) and the zero
      // shift are read directly, other points through the local (comm_type == 1)
      // or general (comm_type == 2) stencil
      T* p_temp = temp_buffer;
      uint64_t block = osites_per_cache_block;

#define fetch_local(obj, _f, site, site_in_block) {			\
	if ((_f)->temporary) {						\
	  obj = coalescedRead(p_temp[(_f)->index * block + site_in_block]); \
	  if ((_f)->adj)						\
	    obj = adj(obj);						\
	} else if ((_f)->zero_point) {					\
	  obj = coalescedRead(fields_v[(_f)->index][site]);		\
	  if ((_f)->adj)						\
	    obj = adj(obj);						\
	} else {							\
	  fetch_shifted(obj, _f, site);					\
	}								\
      }

      if (comm_type == 1) {

	auto sview = general_local_stencil->View(AcceleratorRead);
#define fetch_shifted(obj, _f, site) fetch(obj, (_f)->point, site, fields_v[(_f)->index], (_f)->adj)
#include "matrix_loops.h"
#undef fetch_shifted

      } else {

	// halo exchange of all fields read at non-zero points, then the
	// kernel reads the local fields or the comm buffers
	ASSERT(fields.size() >= gsm->views.size());
	gsm->exchange(fields);
	auto p_gsview = gsm->views.device;
#define fetch_shifted(obj, _f, site) fetch_general(obj, p_gsview[(_f)->index], (_f)->point, site, fields_v[(_f)->index], (_f)->adj)
#include "matrix_loops.h"
#undef fetch_shifted

      }

#undef fetch_local

    } else {

      CGPT_CARTESIAN_STENCIL_HALO_EXCHANGE(T,);

      // now loop
      accelerator_for(ss_block,fields[0].Grid()->oSites() * _npb,T::Nsimd(),{

          uint64_t ss, oblock;
					      
	  MAP_INDEXING(ss, oblock);
	  
	  for (int iblock=0;iblock<_npbs;iblock++) {
	    
	    int i = oblock * _npbs + iblock;
	    
	    obj_t t;
	    
	    const auto _f0 = &p_code[i].factor[0];
	    fetch_cs(stencil_map[_f0->index], t, _f0->point, ss, fields_v[_f0->index], _f0->adj,);
	    
	    for (int j=1;j<p_code[i].size;j++) {
	      obj_t f;
	      const auto _f = &p_code[i].factor[j];
	      fetch_cs(stencil_map[_f->index], f, _f->point, ss, fields_v[_f->index], _f->adj,);
	      t = t * f;
	    }
	    
	    if (!p_code[i].unit_weight)
	      t = p_code[i].weight * t;
	    if (p_code[i].accumulate != -1)
	      t += coalescedRead(fields_v[p_code[i].accumulate][ss]);
	    coalescedWrite(fields_v[p_code[i].target][ss], t);
	  }
	  
	});

      // and cleanup
      CGPT_CARTESIAN_STENCIL_CLEANUP(T,);

    }

    VECTOR_VIEW_CLOSE(fields_v);
  }

  virtual void execute(const std::vector<cgpt_Lattice_base*>& __fields, int fast_osites) {
    PVector<Lattice<T>> fields;
    cgpt_basis_fill(fields,__fields);
    execute(fields, fast_osites);
  }
};

static void cgpt_convert(PyObject* in, cgpt_stencil_matrix_factor_t& out) {
  ASSERT(PyTuple_Check(in));
  ASSERT(PyTuple_Size(in) == 3);
  cgpt_convert(PyTuple_GetItem(in, 0), out.index);
  cgpt_convert(PyTuple_GetItem(in, 1), out.point);
  cgpt_convert(PyTuple_GetItem(in, 2), out.adj);
  out.temporary = 0;
  out.zero_point = 0;
}

static void cgpt_convert(PyObject* in, cgpt_stencil_matrix_code_t& out) {
  ASSERT(PyDict_Check(in));

  out.target = get_int(in, "target");
  out.accumulate = get_int(in, "accumulate");
  out.weight = get_complex(in, "weight");

  cgpt_convert(get_key(in, "factor"), out.factor);
}

// not implemented message
template<typename T>
NotEnableIf<isEndomorphism<T>,cgpt_stencil_matrix_base*>
cgpt_stencil_matrix_create(GridBase* grid, PyObject* _shifts,
			   PyObject* _code, long code_parallel_block_size, long comm_type,
			   PyObject* _temporaries, long osites_per_cache_block) {
  ERR("cgpt_stencil_matrix not implemented for type %s",typeid(T).name());
}

// implemented for endomorphisms
template<typename T>
EnableIf<isEndomorphism<T>,cgpt_stencil_matrix_base*>
cgpt_stencil_matrix_create(GridBase* grid, PyObject* _shifts,
			   PyObject* _code, long code_parallel_block_size, long comm_type,
			   PyObject* _temporaries, long osites_per_cache_block) {

  std::vector<Coordinate> shifts;
  cgpt_convert(_shifts,shifts);

  std::vector<cgpt_stencil_matrix_code_t> code;
  cgpt_convert(_code,code);

  std::vector<int> temporaries;
  cgpt_convert(_temporaries,temporaries);

  return new cgpt_stencil_matrix<T>(grid,shifts,code,code_parallel_block_size, comm_type, temporaries, osites_per_cache_block);
}
