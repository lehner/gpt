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
  virtual void seed(const std::string & s) = 0;
  virtual PyObject* sample(PyObject* target, PyObject* param) = 0;
};

template<typename cgpt_rng_engine>
class cgpt_random_engine : public cgpt_random_engine_base {
 public:
  cgpt_rng_engine cgpt_srng;
  std::map<long,cgpt_rng_engine> cgpt_prng;
  std::vector<long> cgpt_seed;

  cgpt_random_engine() {
  }
  
  virtual ~cgpt_random_engine() {
  }

  virtual void seed(const std::string & seed_str) {
  
    cgpt_seed.resize(0);
    for (auto x : seed_str)
      cgpt_seed.push_back(x);
    
    std::seed_seq seed (cgpt_seed.begin(),cgpt_seed.end());

    cgpt_srng.seed(seed);

  }

  virtual PyObject* sample(PyObject* _target, PyObject* _param) {

    ASSERT(PyDict_Check(_param));
    std::string dist = get_str(_param,"distribution"), precision;
    std::vector<long> shape;
    GridBase* grid = 0;
    int dtype = NPY_COMPLEX128;
    if (PyDict_GetItemString(_param,"shape")) {
      shape = get_long_vec(_param,"shape");
      grid = get_pointer<GridBase>(_param,"grid");
      dtype = infer_numpy_type(get_str(_param,"precision"));
    }

    // always generate in double first regardless of type casting to ensure that numbers are the same up to rounding errors
    // (rng could use random bits to result in different next float/double sampling)
    if (dist == "normal") {
      std::normal_distribution<double> distribution(get_float(_param,"mu"),get_float(_param,"sigma"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid,dtype);
    } else if (dist == "cnormal") {
      cgpt_cnormal_distribution distribution(get_float(_param,"mu"),get_float(_param,"sigma"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid,dtype);
    } else if (dist == "uniform_real") {
      std::uniform_real_distribution<double> distribution(get_float(_param,"min"),get_float(_param,"max"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid,dtype);
    } else if (dist == "uniform_int") {
      std::uniform_int_distribution<int> distribution(get_int(_param,"min"),get_int(_param,"max"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid,dtype);
    } else if (dist == "zn") {
      cgpt_zn_distribution distribution(get_int(_param,"n"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid,dtype);
    } else {
      ERR("Unknown distribution: %s", dist.c_str());
    }

  }

};
