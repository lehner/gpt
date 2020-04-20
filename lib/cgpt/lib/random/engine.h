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
  virtual PyObject* sample(PyObject* target, PyObject* param) = 0;
  virtual double test_U01() = 0;
  virtual uint32_t test_bits() = 0;
};

template<typename cgpt_rng_engine>
class cgpt_random_engine : public cgpt_random_engine_base {
 public:
  cgpt_rng_engine cgpt_srng;
  std::map<long,cgpt_rng_engine> cgpt_prng;
  std::vector<uint64_t> cgpt_seed;

  std::vector<uint64_t> str_to_seed(const std::string & seed_str) {
    std::vector<uint64_t> r;
    for (auto x : seed_str)
      r.push_back(x);
    return r;
  }

  cgpt_random_engine(const std::string & seed_str) : cgpt_seed(str_to_seed(seed_str)), cgpt_srng(str_to_seed(seed_str)) {
  }
  
  virtual ~cgpt_random_engine() {
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
      cgpt_normal_distribution distribution(get_float(_param,"mu"),get_float(_param,"sigma"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid,dtype);
    } else if (dist == "cnormal") {
      cgpt_cnormal_distribution distribution(get_float(_param,"mu"),get_float(_param,"sigma"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid,dtype);
    } else if (dist == "uniform_real") {
      cgpt_uniform_real_distribution distribution(get_float(_param,"min"),get_float(_param,"max"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid,dtype);
    } else if (dist == "uniform_int") {
      cgpt_uniform_int_distribution distribution(get_int(_param,"min"),get_int(_param,"max"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid,dtype);
    } else if (dist == "zn") {
      cgpt_zn_distribution distribution(get_int(_param,"n"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid,dtype);
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

