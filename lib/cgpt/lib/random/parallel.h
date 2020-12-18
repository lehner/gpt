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

template<typename sRNG,typename pRNG>
  void cgpt_random_setup(sRNG & srng, pRNG & prng, GridBase* grid) {

  typedef typename std::remove_reference<decltype(*prng.rng[0])>::type PRNG_t;
  
  // first figure out number of parallel rngs needed for this Grid
  int nd = grid->Nd();
  
  prng.block = 2;
  
  std::vector<long> mpi_dim(nd);
  std::vector<long> cb_dim(nd);
  std::vector<long> block_dim(nd);
  std::vector<long> reduced_dim(nd);

  auto & block = prng.block;
  auto & sites = prng.sites;
  auto & samples = prng.samples;

  if (nd <= 4) {
    for (long i=0;i<nd;i++)
      mpi_dim[i] = true;
  } else if (nd == 5) {
    mpi_dim = {false,true,true,true,true};
  } else {
    ERR("Nd = %ld not yet supported for random interface",nd);
  }

  long blocks = 1;
  sites = (long)grid->_isites * (long)grid->_osites;
  
  for (long j=0;j<nd;j++) {

    // make sure points within a block are always on same node
    // irrespective of MPI setup
    cb_dim[j] = grid->_fdimensions[j] / grid->_gdimensions[j];

    if (mpi_dim[j]) {
      ASSERT(grid->_gdimensions[j] % block == 0);
      ASSERT(grid->_ldimensions[j] % block == 0);
      block_dim[j] = grid->_ldimensions[j] / block;
      reduced_dim[j] = block;
      blocks *= block_dim[j];
    } else {
      block_dim[j] = 1;
      reduced_dim[j] = grid->_ldimensions[j];
    }
  }

  // now we know how many prng's we need ; create hash lookup table
  prng.hash.resize(blocks);
  prng.rng.resize(blocks);
  samples.resize(blocks);
  
  thread_for(idx, blocks, {

      std::vector<long> bcoor;
      Lexicographic::CoorFromIndex(bcoor,idx,block_dim);
      
      long t = 0;
      for (long j=0;j<nd;j++) {
	if (mpi_dim[j]) {
	  int32_t c = bcoor[j] + grid->_lstart[j]/block;
	  ASSERT(c < ((grid->_lend[j]+1)/block));
	  t*=grid->_gdimensions[j] / block;
	  t+=c;
	}
      }

      prng.hash[idx] = t;

      std::vector<uint64_t> _seed = prng.seed;
      _seed.push_back(t);

      prng.rng[idx] = new PRNG_t(_seed);

      samples[idx].resize(sites / blocks);
      
    });

  // now create samples
  thread_region
    {
      Coordinate lcoor(nd);
      std::vector<long> bcoor(nd);
      std::vector<long> rcoor(nd);
      
      thread_for_in_region(idx, sites, {

	  Lexicographic::CoorFromIndex(lcoor,idx,grid->_ldimensions);

	  for (long j=0;j<nd;j++) {
	    if (mpi_dim[j]) {
	      bcoor[j] = lcoor[j] / block;
	      rcoor[j] = lcoor[j] % block;
	    } else {
	      bcoor[j] = 0;
	      rcoor[j] = lcoor[j];
	    }
	  }

	  int block_index, reduced_index;
	  Lexicographic::IndexFromCoor(bcoor,block_index,block_dim);
	  Lexicographic::IndexFromCoor(rcoor,reduced_index,reduced_dim);

	  samples[block_index][reduced_index] = idx;
	});
    }

}

template<typename DIST,typename sRNG,typename pRNG>
PyObject* cgpt_random_sample(DIST & dist,sRNG& srng,pRNG& prng,
			     std::vector<long> & shape,
			     GridBase* grid,int dtype) {

  if (grid) {

    //cgpt_timer t("cgpt_random_sample");

    
    //t("cgpt_random_setup");
    if (prng.hash.size() == 0) {
      cgpt_random_setup(srng, prng, grid);
    }

    //t("push");
    
    //TIME(t4,
    long n = 1;
    std::vector<long> dims;
    dims.push_back(prng.sites);
    for (auto s : shape) {
      n *= s;
      dims.push_back(s);
    }

    //t("array");
    // all the previous effort allows the prng to act in parallel
    PyArrayObject* a = (PyArrayObject*)PyArray_SimpleNew((int)dims.size(), &dims[0], dtype);
    auto & samples = prng.samples;

    //t("fill");

    double t0 = cgpt_time();
    if (dtype == NPY_COMPLEX64) {
      ComplexF* d = (ComplexF*)PyArray_DATA(a);
      thread_for(i, samples.size(), {
	  auto & r = prng.rng[i];
	  auto & s = samples[i];
	  for (long x=0;x<s.size();x++) for (long j=0;j<n;j++) d[n*s[x]+j] = (ComplexF)dist(*r);
	});
    } else if (dtype == NPY_COMPLEX128) {
      ComplexD* d = (ComplexD*)PyArray_DATA(a);
      thread_for(i, samples.size(), {
	  auto & r = prng.rng[i];
	  auto & s = samples[i];
	  for (long x=0;x<s.size();x++) for (long j=0;j<n;j++) d[n*s[x]+j] = (ComplexD)dist(*r);
	});
    } else {
      ERR("Unknown dtype");
    }
    double t1 = cgpt_time();
    //std::cout << GridLogMessage << "Fill: " << t1- t0 << std::endl;

    //t.report();
    
    //	 );

    return (PyObject*)a;

  } else {
    ComplexD val = (ComplexD)dist(srng);
    return PyComplex_FromDoubles(val.real(),val.imag());
  }

}
