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
static void cgpt_random_to_hash(PyArrayObject* coordinates,std::vector<long>& hashes,GridBase* grid) {
  ASSERT(PyArray_NDIM(coordinates) == 2);
  long* tdim = PyArray_DIMS(coordinates);
  long nc = tdim[0];
  long nd = tdim[1];
  std::vector<bool> mpi_dim(nd);
  if (nd == 4) {
    mpi_dim = {true,true,true,true};
  } else if (nd == 5) {
    mpi_dim = {false,true,true,true,true};
  } else {
    ERR("Nd = %ld not yet supported for random interface",nd);
  }

  const int block = 2;
  ASSERT(nd == grid->Nd());

  ASSERT(PyArray_TYPE(coordinates)==NPY_INT32);
  int32_t* coor = (int32_t*)PyArray_DATA(coordinates);

  std::vector<int> cb_dim(nd);
  for (long j=0;j<nd;j++) {
    if (mpi_dim[j]) {
      ASSERT(grid->_gdimensions[j] % block == 0);
      ASSERT(grid->_ldimensions[j] % block == 0); 
    }
    // make sure points within a block are always on same node
    // irrespective of MPI setup
    cb_dim[j] = grid->_fdimensions[j] / grid->_gdimensions[j];
  }

  hashes.resize(nc);
  thread_for(i, nc, {
      long t = 0;
      for (long j=0;j<nd;j++) {
	if (mpi_dim[j]) {
	  int32_t c = coor[nd*i+j] / cb_dim[j] / block;
	  ASSERT((c >= (grid->_lstart[j]/block)) && (c < ((grid->_lend[j]+1)/block)));
	  t*=grid->_gdimensions[j] / block;
	  t+=c;
	}
      }
      hashes[i] = t;
    });
}

template<typename T>
void cgpt_hash_offsets(std::vector<T>& h, std::map<T,std::vector<long>>& u) {
  // This is a candidate for optimization; for ranlux48 not dominant but not negligible
  for (long off=0;off<h.size();off++)
    u[h[off]].push_back(off);
}

template<typename T>
void cgpt_hash_unique(std::map<T,std::vector<long>>& u, std::vector<T>& h) {
  for (auto&k : u)
    h.push_back(k.first);
  //std::cout << GridLogMessage << h.size() << " unique parallel RNGs" << std::endl;
}

template<typename sRNG,typename pRNG>
  void cgpt_random_setup(std::vector<long>& h,sRNG & srng,std::map<long,pRNG> & prng,std::vector<uint64_t> & seed) {
  std::vector<long> need;
  for (auto x : h) {
    auto p = prng.find(x);
    if (p == prng.end()) {
      need.push_back(x);
    }
  }

  //std::cout << GridLogMessage << need.size() << " new parallel RNGs" << std::endl;

  thread_for(i, need.size(), {

      long x = need[i];
      std::vector<uint64_t> _seed = seed;
      _seed.push_back(x);

      auto pr = pRNG(_seed);
      
      thread_critical
	{
	  prng.insert({x,pr});
	}
    });
}


template<typename DIST,typename sRNG,typename pRNG>
  PyObject* cgpt_random_sample(DIST & dist,PyObject* _target,sRNG& srng,pRNG& prng,
			       std::vector<long> & shape,std::vector<uint64_t> & seed,
			       GridBase* grid,int dtype) {

  if (PyArray_Check(_target)) {

    //TIME(t0,

    std::vector<long> hashes;
    cgpt_random_to_hash((PyArrayObject*)_target,hashes,grid);

    //	 );

    //TIME(t1,
    std::map<long,std::vector<long>> hash_offsets;
    cgpt_hash_offsets(hashes,hash_offsets);
    //	 );

    //TIME(t2,
    std::vector<long> unique_hashes;
    cgpt_hash_unique(hash_offsets,unique_hashes);
    //	 );

    //TIME(t3,
    cgpt_random_setup(unique_hashes,srng,prng,seed);
    //	 );

    //TIME(t4,
    long n = 1;
    std::vector<long> dims;
    dims.push_back(hashes.size());
    for (auto s : shape) {
      n *= s;
      dims.push_back(s);
    }

    // all the previous effort allows the prng to act in parallel
    PyArrayObject* a = (PyArrayObject*)PyArray_SimpleNew((int)dims.size(), &dims[0], dtype);
    if (dtype == NPY_COMPLEX64) {
      ComplexF* d = (ComplexF*)PyArray_DATA(a);
      thread_for(i, unique_hashes.size(), {
	  auto h = unique_hashes[i];
	  auto & uhi = hash_offsets[h];
	  for (auto & x : uhi) for (long j=0;j<n;j++) d[n*x+j] = (ComplexF)dist(prng.find(h)->second);
	});
    } else if (dtype == NPY_COMPLEX128) {
      ComplexD* d = (ComplexD*)PyArray_DATA(a);
      thread_for(i, unique_hashes.size(), {
	  auto h = unique_hashes[i];
	  auto & uhi = hash_offsets[h];
	  for (auto & x : uhi) for (long j=0;j<n;j++) d[n*x+j] = dist(prng.find(h)->second);
	});
    } else {
      ERR("Unknown dtype");
    }
    //	 );

    //std::cout << GridLogMessage << "Timing: " << t0 << ", " << t1 << ", " << t2 << ", " << t3 << ", " << t4 << std::endl;
    return (PyObject*)a;

  } else if (_target == Py_None) {
    ComplexD val = (ComplexD)dist(srng);
    return PyComplex_FromDoubles(val.real(),val.imag());
  } else {
    ERR("_target type not implemented");
  }

}
