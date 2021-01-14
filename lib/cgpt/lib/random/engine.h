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
class cgpt_random_engine_base {
public:
  virtual ~cgpt_random_engine_base() { };
  virtual PyObject* sample(PyObject* param) = 0;
  virtual double test_U01() = 0;
  virtual uint32_t test_bits() = 0;
};

template<typename cgpt_rng_engine>
class cgpt_random_engine : public cgpt_random_engine_base {
 public:
  std::string _seed_str;
  cgpt_rng_engine cgpt_srng;

  struct index_t {
    int i,o; // adjust once grid adjusts
  };
  
  struct prng_t {
    std::vector<cgpt_rng_engine*> rng;
    std::vector<long> hash;
    std::vector<uint64_t> seed;
    std::vector< std::vector< index_t > > samples;
    long block, sites;
    std::string grid_tag;
  };
  
  std::map<GridBase*,prng_t> cgpt_prng;

  std::vector<uint64_t> str_to_seed(const std::string & seed_str) {
    std::vector<uint64_t> r;
    for (auto x : seed_str)
      r.push_back(x);
    return r;
  }

  cgpt_random_engine(const std::string & seed_str) : _seed_str(seed_str), cgpt_srng(str_to_seed(seed_str)) {
  }
  
  virtual ~cgpt_random_engine() {
    for (auto & x : cgpt_prng)
      for (auto & y : x.second.rng)
	delete y;
  }

  virtual PyObject* sample(PyObject* _param) {

    double t0 = cgpt_time();
    
    ASSERT(PyDict_Check(_param));
    std::string dist = get_str(_param,"distribution");
    GridBase* grid = 0;
    std::vector<cgpt_Lattice_base*> lattices;
    long n_virtual;
    PyObject* _lattices = PyDict_GetItemString(_param,"lattices");
    if (_lattices) {
      n_virtual = cgpt_basis_fill(lattices,_lattices);
      ASSERT(lattices.size() > 0);
      grid = lattices[0]->get_grid();
      for (size_t i=1;i<lattices.size();i++)
	ASSERT(grid == lattices[i]->get_grid());
    }

    prng_t & prng = cgpt_prng[grid];
    if (prng.grid_tag.size() == 0) {
      prng.grid_tag = cgpt_grid_cache_tag[grid];
    }
    if (prng.grid_tag != cgpt_grid_cache_tag[grid]) {
      // Grid changed! clear cache
      prng = prng_t();
    }
    std::vector<uint64_t> & seed = prng.seed;
    if (seed.size() == 0) {
      seed = str_to_seed(_seed_str);
      if (grid) {
	for (auto x : grid->_fdimensions)
	  seed.push_back(x);
	for (auto x : grid->_gdimensions)
	  seed.push_back(x);
      }
    }

    double t1 = cgpt_time();
    //std::cout << GridLogMessage << "prep " << t1-t0 << std::endl;

    // always generate in double first regardless of type casting to ensure that numbers are the same up to rounding errors
    // (rng could use random bits to result in different next float/double sampling)
    if (dist == "normal") {
      cgpt_normal_distribution distribution(get_float(_param,"mu"),get_float(_param,"sigma"));
      return cgpt_random_sample(distribution,cgpt_srng,prng,lattices,n_virtual);
    } else if (dist == "cnormal") {
      cgpt_cnormal_distribution distribution(get_float(_param,"mu"),get_float(_param,"sigma"));
      return cgpt_random_sample(distribution,cgpt_srng,prng,lattices,n_virtual);
    } else if (dist == "uniform_real") {
      cgpt_uniform_real_distribution distribution(get_float(_param,"min"),get_float(_param,"max"));
      return cgpt_random_sample(distribution,cgpt_srng,prng,lattices,n_virtual);
    } else if (dist == "uniform_int") {
      cgpt_uniform_int_distribution distribution(get_int(_param,"min"),get_int(_param,"max"));
      return cgpt_random_sample(distribution,cgpt_srng,prng,lattices,n_virtual);
    } else if (dist == "zn") {
      cgpt_zn_distribution distribution(get_int(_param,"n"));
      return cgpt_random_sample(distribution,cgpt_srng,prng,lattices,n_virtual);
    } else {
      ERR("Unknown distribution: %s", dist.c_str());
    }

  }

  virtual double test_U01() {
    return cgpt_srng.get_double();
  }

  virtual uint32_t test_bits() {
    return cgpt_srng.get_uint32_t();
  }

};

