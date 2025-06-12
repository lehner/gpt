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


    This code is based on original Grid code.
*/

#ifdef GRID_HAS_ACCELERATOR
#define rankInnerProduct rankInnerProductGPU
#else
#define rankInnerProduct rankInnerProductCpu
#endif  

#if defined(GRID_SYCL)

#define accelerator_threads_sync() item.barrier(sycl::access::fence_space::local_space);

#define SYCL_SIMD_WIDTH 16

#define accelerator_forNB_with_shared( iter1, num1, nsimd, shared_type, shared_name, ... ) { \
    acceleratorFenceComputeStream();					\
    theGridAccelerator->submit([&](sycl::handler &cgh) {		\
      unsigned long unum1 = num1;					\
      unsigned long unum1_divisible_by = ((unum1 + SYCL_SIMD_WIDTH - 1) / SYCL_SIMD_WIDTH) * SYCL_SIMD_WIDTH; \
      sycl::range<3> local {1,1,nsimd};				\
      sycl::range<3> global{unum1_divisible_by,1,nsimd};		\
      sycl::local_accessor<shared_type> shm_acc(sycl::range<1>(nsimd), cgh); \
      if (unum1 != unum1_divisible_by) {				\
	cgh.parallel_for(						\
			 sycl::nd_range<3>(global,local),		\
			 [=] (sycl::nd_item<3> item)		\
			 [[intel::reqd_sub_group_size(SYCL_SIMD_WIDTH)]] \
			 {						\
			   auto iter1    = item.get_global_id(0);	\
			   auto lane     = item.get_global_id(2);	\
			   shared_type* shared_name = shm_acc.get_pointer(); \
			   { if (iter1 < unum1) { __VA_ARGS__ } };	\
			 });						\
      } else {								\
	cgh.parallel_for(						\
			 sycl::nd_range<3>(global,local),		\
			 [=] (sycl::nd_item<3> item)		\
			 [[intel::reqd_sub_group_size(SYCL_SIMD_WIDTH)]] \
			 {						\
			   auto iter1    = item.get_global_id(0);	\
			   auto lane     = item.get_global_id(2);	\
			   shared_type* shared_name = shm_acc.get_pointer(); \
			   { __VA_ARGS__ };				\
			 });						\
      }									\
    });									\
  }


#else

#define accelerator_threads_sync acceleratorSynchronise

#ifdef GRID_SIMT
#define accelerator_shared __shared__
#else
#define accelerator_shared
#endif

#define accelerator_forNB_with_shared( iter1, num1, nsimd, shared_type, shared_name, ... ) { \
    auto _stored_nt = acceleratorThreads();				\
    acceleratorThreads(1);						\
    accelerator_forNB(iter1, num1, nsimd, {accelerator_shared shared_type shared_name[nsimd]; __VA_ARGS__}); \
    acceleratorThreads(_stored_nt);					\
  }

#endif

template<int n_per_thread, int n_coalesce, typename T>
inline void rankInnerProductGPU_reduce(uint64_t n_total, ComplexD* result, uint64_t n_left, uint64_t n_right, uint64_t n_virtual, uint64_t n_block,
				       T left_v, T right_v) {

  typedef typename std::remove_reference<decltype(left_v[0][0])>::type::scalar_type scalar_type;

  ASSERT(n_total % n_per_thread == 0);
  const uint64_t n_reduced = n_total / n_per_thread;

  const uint64_t n_outer = (n_reduced + n_coalesce - 1) / n_coalesce;

  const uint64_t n_inner = n_left * n_right * n_block;

  uint64_t n_total_stride = 1;
  for (uint64_t n_stride=n_outer * n_virtual;n_stride > 1;n_stride = (n_stride + n_coalesce - 1) / n_coalesce) {
    n_total_stride += n_stride;
  }

  Vector<ComplexD> inner_tmp;
  inner_tmp.resize(n_total_stride * n_inner);
  
  auto inner_tmp_v = &inner_tmp[0];

  Timer("rip: loop");
  {
    for (int i=0;i<n_virtual;i++) {

      for (int kb=0;kb<n_block;kb++) {
	uint64_t l0 = kb*n_virtual*n_left;
	uint64_t r0 = kb*n_virtual*n_right;
	uint64_t t0 = kb*n_left*n_right;

	for (int kl=0;kl<n_left;kl++) {
	  scalar_type* a = (scalar_type*)left_v[l0 + kl*n_virtual + i];

	  accelerator_forNB_with_shared(work, n_outer, n_coalesce, ComplexD, vc, {
	    
	      int lane = acceleratorSIMTlane(n_coalesce);
	      uint64_t idx0 = work * n_per_thread * n_coalesce;

	      // avoid inner *if* for as long as possible
	      bool do_all = (lane + (n_per_thread - 1)*n_coalesce + idx0) < n_total;
	      
	      for (int kr=0;kr<n_right;kr++) {
		int idx_result = t0 + kl * n_right + kr;
	      
		scalar_type* b = (scalar_type*)right_v[r0 + kr*n_virtual + i];
		ComplexD* c = (ComplexD*)&inner_tmp_v[(idx_result * n_virtual + i) * n_outer];

		ComplexD v = 0.0;

		if (do_all) {
		  for (int j=0;j<n_per_thread;j++) {
		    uint64_t idx = lane + j*n_coalesce + idx0;
		    v += (ComplexD)conjugate(a[idx]) * (ComplexD)b[idx];
		  }
		} else {
		  for (int j=0;j<n_per_thread;j++) {
		    uint64_t idx = lane + j*n_coalesce + idx0;
		    if (idx < n_total)
		      v += (ComplexD)conjugate(a[idx]) * (ComplexD)b[idx];
		  }
		}	      
	    
		vc[lane] = v;

		accelerator_threads_sync();
		
		if (lane == 0) {
		  v = 0.0;
		  for (int j=0;j<n_coalesce;j++)
		    v+=vc[j];
		  c[work] = v;
		}
		
		accelerator_threads_sync();
	      }
	    });
	}
      }
    }
    // now reduce from n_outer -> n_outer / n_coalesce
    uint64_t n_stride = n_outer * n_virtual;
    ComplexD* src_base = &inner_tmp_v[0];
    while (n_stride > 1) {

      ComplexD* dst_base = &src_base[n_inner * n_stride];
      
      uint64_t n_stride_prime = (n_stride + n_coalesce - 1) / n_coalesce;

      accelerator_forNB_with_shared(work, n_stride_prime, n_coalesce, ComplexD, vc, {
	    
	  int lane = acceleratorSIMTlane(n_coalesce);

	  uint64_t idx0 = work * n_coalesce + lane;
	  bool active = idx0 < n_stride;
	  
	  for (int i=0;i<n_inner;i++) {

	    ComplexD v = 0.0;

	    if (active) {
	      v = src_base[i*n_stride + idx0];
	    }
	    
	    vc[lane] = v;
	    
	    accelerator_threads_sync();
	    
	    if (lane == 0) {
	      v = 0.0;
	      for (int ii=0;ii<n_coalesce;ii++)
		v+=vc[ii];
	      dst_base[i*n_stride_prime + work] = v;
	    }
	    
	    accelerator_threads_sync();
	  }
	});
      
      n_stride = n_stride_prime;
      src_base = dst_base;
    }
    accelerator_barrier();
    // results are now here:
    for (int i=0;i<n_inner;i++) {
      result[i] = src_base[i];
    }
  }
}

class tensor_ComplexD : public ComplexD {
public:
  typedef tensor_ComplexD scalar_objectD;
  typedef tensor_ComplexD scalar_object;
  typedef tensor_ComplexD scalar_type;
  typedef tensor_ComplexD vector_type;
  accelerator tensor_ComplexD() : ComplexD() {};
  accelerator_inline tensor_ComplexD(const Zero &z) { *(ComplexD*)this = 0.0; };
  static accelerator_inline constexpr int Nsimd(void) { return 1; } 
};

template<class vobj>
inline void rankInnerProductGPU(ComplexD* result, 
				PVector<Lattice<vobj>> &multi_left,
				PVector<Lattice<vobj>> &multi_right,
				size_t n_virtual,
				size_t n_block)
{
  GridBase *grid = multi_left[0].Grid();

  size_t n_v_block = n_virtual * n_block;
  assert(multi_left.size() % n_v_block == 0);
  assert(multi_right.size() % n_v_block == 0);

  const uint64_t n_left = multi_left.size() / n_v_block;
  const uint64_t n_right = multi_right.size() / n_v_block;

  constexpr int n_reduction_min = GridTypeMapper<vobj>::count;
  const uint64_t n_total = grid->oSites() * n_reduction_min * vobj::Nsimd();

  Timer("rip: view");
  VECTOR_VIEW_OPEN(multi_left,left_v,AcceleratorRead);
  VECTOR_VIEW_OPEN(multi_right,right_v,AcceleratorRead);

  constexpr uint64_t n_coalesce = 32; // good idea to align this with warp size
#define GPU_IP_N(n_per_thread) if (n_total % n_per_thread == 0) { \
    rankInnerProductGPU_reduce<n_per_thread,n_coalesce>(n_total, result, n_left, n_right, n_virtual, n_block, left_v, right_v); \
  } else
  //GPU_IP_N(32)
  GPU_IP_N(16)
  GPU_IP_N(8)
  GPU_IP_N(4)
  GPU_IP_N(2)
  {
    rankInnerProductGPU_reduce<n_reduction_min,vobj::Nsimd()>(n_total, result, n_left, n_right, n_virtual, n_block, left_v, right_v);
  }
  
  Timer("rip: view");
  VECTOR_VIEW_CLOSE(left_v);
  VECTOR_VIEW_CLOSE(right_v);

  Timer();
}


template<class vobj>
inline void rankInnerProductCpu(ComplexD* result, 
				PVector<Lattice<vobj>> &multi_left,
				PVector<Lattice<vobj>> &multi_right,
				size_t n_virtual,
				size_t n_block)
{
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;
  typedef typename GridTypeMapper<vector_type>::DoublePrecision2 vector_typeD2;
  
  GridBase *grid = multi_left[0].Grid();

  size_t n_v_block = n_virtual * n_block;
  assert(multi_left.size() % n_v_block == 0);
  assert(multi_right.size() % n_v_block == 0);

  const uint64_t n_left = multi_left.size() / n_v_block;
  const uint64_t n_right = multi_right.size() / n_v_block;
  const uint64_t words_per_osite = sizeof(vobj) / sizeof(vector_type);
  const uint64_t words = grid->oSites() * words_per_osite;
  const uint64_t max_parallel = thread_max();

  VECTOR_VIEW_OPEN(multi_left,left_v,CpuRead);
  VECTOR_VIEW_OPEN(multi_right,right_v,CpuRead);
  
  {
    size_t stride = n_left*n_right*n_block;
    AlignedVector<ComplexD> all_thread_sum_reduce(max_parallel * stride);
    
    thread_region
      {
	ComplexD * thread_sum_reduce = &all_thread_sum_reduce[thread_num()*stride];
	thread_for_in_region(i, all_thread_sum_reduce.size(), {
	    all_thread_sum_reduce[i] = 0.0;
	  });

	thread_for_in_region( w, words, {
	    for (uint64_t kb=0;kb<n_block;kb++) {
	      uint64_t l0 = kb*n_virtual*n_left;
	      uint64_t r0 = kb*n_virtual*n_right;
	      uint64_t t0 = kb*n_left*n_right;
	      
	      for (uint64_t kl=0;kl<n_left;kl++) {
		for (uint64_t kr=0;kr<n_right;kr++) {
		  ComplexD s = 0.0;
		  for (size_t i=0;i<n_virtual;i++) {
		    vector_type* l = (vector_type*)&left_v[l0 + kl*n_virtual + i][0];
		    vector_type* r = (vector_type*)&right_v[r0 + kr*n_virtual + i][0];
		    s += Reduce(innerProductD(l[w],r[w]));
		  }
		  thread_sum_reduce[t0 + kl * n_right + kr] += s;
		}
	      }
	    }
	  });

	thread_for_in_region( i, stride, {
	    result[i] = 0.0;
	    for (uint64_t j=0;j<max_parallel;j++) {
	      ComplexD * thread_sum_reduce = &all_thread_sum_reduce[j*stride];
	      result[i] += thread_sum_reduce[i];
	    }
	  });

      }

  }

  VECTOR_VIEW_CLOSE(left_v);
  VECTOR_VIEW_CLOSE(right_v);
}
