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
template<class vobj, class stype>
inline void sumD_cpu(stype* ret, const vobj *arg, Integer osites, Integer nvec)
{
  const int nthread = thread_max();

  AlignedVector<stype> sum(nthread * nvec);

  thread_region
    {
      int t = thread_num();

      for (Integer i=0;i<nvec;i++) {
	vobj vs=Zero();
	thread_for_in_region(idx, osites, {
	    vs += arg[osites*i + idx];
	  });
	sum[t * nvec + i] = Reduce(vs);
      }
    }


  thread_for(i,nvec, {
      stype r = 0.0;
      for (int j=0;j<nthread;j++)
	r += sum[j*nvec + i];
      ret[i] = r;
    });

}

template<class vobj, class stype>
inline void sumD(stype* res, const vobj *arg, Integer osites, Integer nvec)
{
#if defined(GRID_CUDA)||defined(GRID_HIP)
  sumD_gpu(res,arg,osites,nvec);
#else
  sumD_cpu(res,arg,osites,nvec);
#endif  
}

template<class vobj>
inline void rankInnerProduct(ComplexD* result, 
			     PVector<Lattice<vobj>> &multi_left,
			     PVector<Lattice<vobj>> &multi_right,
			     size_t n_virtual)
{
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  GridBase *grid = multi_left[0].Grid();

  assert(multi_left.size() % n_virtual == 0);
  assert(multi_right.size() % n_virtual == 0);

  const uint64_t n_left = multi_left.size() / n_virtual;
  const uint64_t n_right = multi_right.size() / n_virtual;
  const uint64_t sites = grid->oSites();

  typedef decltype(innerProductD(vobj(),vobj())) inner_t;
  VECTOR_VIEW_OPEN(multi_left,left_v,AcceleratorRead);
  VECTOR_VIEW_OPEN(multi_right,right_v,AcceleratorRead);

  Vector<inner_t> inner_tmp(sites * n_left * n_right);
  auto inner_tmp_v = &inner_tmp[0];

  {
    accelerator_for( work, sites * n_left * n_right, 1,{
	uint64_t _work = work;
#ifdef GRID_HAS_ACCELERATOR
	uint64_t ss = _work % sites; _work /= sites;
	uint64_t kr = _work % n_right; _work /= n_right;
	uint64_t kl = _work % n_left;// _work /= n_left;
#else
	uint64_t kr = _work % n_right; _work /= n_right;
	uint64_t kl = _work % n_left; _work /= n_left;
	uint64_t ss = _work % sites; //_work /= sites;
#endif
	inner_t vs;
	zeroit(vs);
	for (size_t i=0;i<n_virtual;i++)
	  vs+=innerProductD(left_v[kl*n_virtual+i][ss],right_v[kr*n_virtual+i][ss]);
	inner_tmp_v[ ss + sites * ( kl * n_right + kr ) ] = vs;
    });
  }

  VECTOR_VIEW_CLOSE(left_v);
  VECTOR_VIEW_CLOSE(right_v);

  sumD(result,inner_tmp_v,(Integer)sites,(Integer)(n_left*n_right));
}

template<class vobj>
inline void rankInnerProductCpu(ComplexD* result, 
				PVector<Lattice<vobj>> &multi_left,
				PVector<Lattice<vobj>> &multi_right,
				size_t n_virtual)
{
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;
  typedef typename GridTypeMapper<vector_type>::DoublePrecision2 vector_typeD2;
  
  GridBase *grid = multi_left[0].Grid();

  assert(multi_left.size() % n_virtual == 0);
  assert(multi_right.size() % n_virtual == 0);

  const uint64_t n_left = multi_left.size() / n_virtual;
  const uint64_t n_right = multi_right.size() / n_virtual;
  const uint64_t words_per_osite = sizeof(vobj) / sizeof(vector_type);
  const uint64_t words = grid->oSites() * words_per_osite;
  const uint64_t max_parallel = thread_max();

  VECTOR_VIEW_OPEN(multi_left,left_v,CpuRead);
  VECTOR_VIEW_OPEN(multi_right,right_v,CpuRead);
  
  {
    AlignedVector<ComplexD> all_thread_sum_reduce(max_parallel * n_left*n_right);
    
    thread_region
      {
	ComplexD * thread_sum_reduce = &all_thread_sum_reduce[thread_num()*n_left*n_right];
	thread_for_in_region(i, all_thread_sum_reduce.size(), {
	    all_thread_sum_reduce[i] = 0.0;
	  });

	thread_for_in_region( w, words, {
	    for (uint64_t kl=0;kl<n_left;kl++) {
	      for (uint64_t kr=0;kr<n_right;kr++) {
		ComplexD s = 0.0;
		for (size_t i=0;i<n_virtual;i++) {
		  vector_type* l = (vector_type*)&left_v[kl*n_virtual + i][0];
		  vector_type* r = (vector_type*)&right_v[kr*n_virtual + i][0];
		  s += Reduce(innerProductD(l[w],r[w]));
		}
		thread_sum_reduce[kl * n_right + kr] += s;
	      }
	    }
	  });

	thread_for_in_region( i, n_left*n_right, {
	    result[i] = 0.0;
	    for (uint64_t j=0;j<max_parallel;j++) {
	      ComplexD * thread_sum_reduce = &all_thread_sum_reduce[j*n_left*n_right];
	      result[i] += thread_sum_reduce[i];
	    }
	  });

      }

  }

  VECTOR_VIEW_CLOSE(left_v);
  VECTOR_VIEW_CLOSE(right_v);
}
