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

	    lcoor[j] *= grid->_fdimensions[j] / grid->_gdimensions[j];
	  }

	  auto oidx = grid->oIndex(lcoor);
	  auto iidx = grid->iIndex(lcoor);

	  int block_index, reduced_index;
	  Lexicographic::IndexFromCoor(bcoor,block_index,block_dim);
	  Lexicographic::IndexFromCoor(rcoor,reduced_index,reduced_dim);

	  samples[block_index][reduced_index] = { iidx, oidx };
	});
    }

}

template<typename prng_t, typename dist_t, typename complex_t, typename tensor_t>
void cgpt_random_sample_lattice(prng_t & prng, dist_t & dist,
				size_t blocks, size_t n_per_block, size_t n_per_site,
				std::vector<cgpt_Lattice_base*> & lattices,
				long n_virtual, size_t osite_stride, size_t isite_stride,
				size_t obj_stride, tensor_t & t,
				complex_t c) {

  auto & samples = prng.samples;
  auto & rng = prng.rng;

  ASSERT(n_virtual == (long)lattices.size());

  AlignedVector<complex_t> data(blocks * n_per_block);

  //double t0 = cgpt_time();
  thread_for(idx, blocks, {
      auto & pr = rng[idx];
      for (size_t i=0;i<n_per_block;i++)
  	data[idx * n_per_block + i] = (complex_t)dist(*pr);
  });
  //double t1 = cgpt_time();
  //double gb = blocks * n_per_block * sizeof(complex_t) / 1024./1024./1024.;
  //std::cout << GridLogMessage << "Raw generation speed: " << gb / (t1-t0) << " GB/s" << std::endl;

  std::vector<PyObject*> view(lattices.size());
  for (size_t i=0;i<view.size();i++)
    view[i] = lattices[i]->memory_view(mt_host);

  std::vector<complex_t*> p(view.size());
  for (size_t i=0;i<view.size();i++)
    p[i] = (complex_t*)PyMemoryView_GET_BUFFER(view[i])->buf;

  //t0 = cgpt_time();
  thread_for(idx, blocks, {
      auto & pr = rng[idx];
      auto & s = samples[idx];
      for (size_t i=0;i<s.size();i++) {
	auto ii = s[i].i;
	auto oo = s[i].o;
	size_t lattice_site_index = oo * osite_stride + ii * isite_stride;
	size_t data_site_index = idx * n_per_block + i*n_per_site;
	for (size_t j=0;j<n_per_site;j++) {
	  p[t[j].idx_lat][lattice_site_index + t[j].idx_obj * obj_stride] = data[data_site_index + j];
	}
      }
    });
  //t1 = cgpt_time();
  //std::cout << GridLogMessage << "Raw assignment speed: " << gb / (t1-t0) << " GB/s" << std::endl;

  for (size_t i=0;i<view.size();i++)
    Py_DECREF(view[i]); // close views
}

template<typename DIST,typename sRNG,typename pRNG>
PyObject* cgpt_random_sample(DIST & dist,sRNG& srng,pRNG& prng,
			     std::vector<cgpt_Lattice_base*> & lattices,
			     long n_virtual) {
  
  if (lattices.size() > 0) {

    GridBase* grid = lattices[0]->get_grid();

    //cgpt_timer t("cgpt_random_sample");

    
    //t("cgpt_random_setup");
    if (prng.hash.size() == 0) {
      cgpt_random_setup(srng, prng, grid);
    }

    //t("fill");

    size_t n_per_site = 0; // number of complex numbers per site
    size_t complex_size = 0;
    size_t sites = (size_t)grid->_osites * (size_t)grid->_isites;
    size_t blocks = prng.rng.size();
    size_t osite_stride, isite_stride, obj_stride;
    int singlet_rank;

    struct tensor_t {
      int idx_lat;
      int idx_obj;
    };
    
    ASSERT(sites % blocks == 0);
    for (auto & l : lattices) {
      long Nsimd, word, simd_word;
      std::vector<long> ishape;
      l->describe_data_layout(Nsimd, word, simd_word);
      l->describe_data_shape(ishape);
      long l_n_per_site = word / simd_word;
      n_per_site += (size_t)l_n_per_site;
      if (complex_size) {
	ASSERT(complex_size == (size_t)simd_word);
	ASSERT(osite_stride == (size_t)(word * Nsimd / simd_word));
	ASSERT(obj_stride == (size_t)Nsimd);
	ASSERT(singlet_rank == l->singlet_rank());
      } else {
	complex_size = (size_t)simd_word;
	osite_stride = word * Nsimd / simd_word;
	obj_stride = Nsimd;
	singlet_rank = l->singlet_rank();
	isite_stride = 1;
      }
      //std::cout << GridLogMessage << "Shape: " <<ishape << std::endl;
    }
    size_t n_per_block = n_per_site * (sites / blocks);

    std::vector<tensor_t> t(n_per_site);

    size_t v_dim = size_to_singlet_dim(singlet_rank, (int)lattices.size());
    size_t t_dim = singlet_rank == 0 ? n_per_site : size_to_singlet_dim(singlet_rank, (int)n_per_site);
    
    std::vector<int> vdim(singlet_rank == 0 ? 1 : singlet_rank, v_dim);
    std::vector<int> tdim(singlet_rank == 0 ? 1 : singlet_rank, t_dim);
    std::vector<int> ldim(singlet_rank == 0 ? 1 : singlet_rank, t_dim / v_dim);

    /*
      std::cout << GridLogMessage << lattices[0]->type() << " : " <<
      " n_per_site = " << n_per_site <<
      " osite_stride = " << osite_stride <<
      " obj_stride = " << obj_stride <<
      " v_dim = " << v_dim <<
      " t_dim = " << t_dim <<
      " singlet_rank = " << singlet_rank <<
      std::endl;
    */
    
    std::vector<int> vcoor(vdim.size()), tcoor(vdim.size()), lcoor(vdim.size());
    for (int i=0;i<(int)n_per_site;i++) {
      Lexicographic::CoorFromIndexReversed(tcoor,i,tdim);

      for (size_t l=0;l<vdim.size();l++) {
	vcoor[l] = tcoor[l] / ldim[l]; // coordinate of virtual lattice
	lcoor[l] = tcoor[l] - vcoor[l] * ldim[l]; // coordinate within virtual lattice
      }
      
      int iidx, lidx;
      Lexicographic::IndexFromCoor(vcoor,iidx,vdim);
      Lexicographic::IndexFromCoorReversed(lcoor,lidx,ldim);

      t[i].idx_lat = iidx;
      t[i].idx_obj = lidx;

      //std::cout << GridLogMessage << i << " is on lat " << iidx << " and obj_idx " << lidx << std::endl;
    }

    // complex_size dependent code below
    //double t0 = cgpt_time();

    if (complex_size == sizeof(ComplexF)) {
      cgpt_random_sample_lattice(prng, dist, blocks, n_per_block, n_per_site, lattices, n_virtual, osite_stride, isite_stride, obj_stride, t, ComplexF());
    } else if (complex_size == sizeof(ComplexD)) {
      cgpt_random_sample_lattice(prng, dist, blocks, n_per_block, n_per_site, lattices, n_virtual, osite_stride, isite_stride, obj_stride, t, ComplexD());
    } else {
      ERR("Unknown sizeof(complex) = %ld", (long)complex_size);
    }
    
    //double t1 = cgpt_time();

    //double speed = n_per_site * sites * sizeof(ComplexD) / 1024./1024./1024. / (t1 - t0);
    //std::cout << GridLogMessage << "Speed: " << speed << " GB/s" << std::endl;

    //t.report();
    
    //	 );

    return PyLong_FromLong(0);

  } else {
    ComplexD val = (ComplexD)dist(srng);
    return PyComplex_FromDoubles(val.real(),val.imag());
  }

}
