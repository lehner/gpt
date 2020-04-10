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
#include "lib.h"
#include "random.h"

/*
  rng[hash[pos]]

  hash[pos] : divide lattice into 2^nd blocks/hashes
              need to make sure that no two ranks have same hash

	      for 32^4 local volume, this means 65536 local hashes

	      when do we send all hashes do root to check for duplicates
*/
typedef std::ranlux48 cgpt_rng_engine;
static cgpt_rng_engine cgpt_srng;
static std::map<long,cgpt_rng_engine> cgpt_prng;
static std::vector<long> cgpt_seed;

EXPORT(random_seed,{

    PyObject* _seed_str;
    if (!PyArg_ParseTuple(args, "O", &_seed_str)) {
      return NULL;
    }

    std::string seed_str;
    cgpt_convert(_seed_str,seed_str);
    cgpt_seed.resize(0);
    for (auto x : seed_str)
      cgpt_seed.push_back(x);

    std::seed_seq seed (cgpt_seed.begin(),cgpt_seed.end());

    cgpt_srng.seed(seed);

    return PyLong_FromLong(0);

  });

EXPORT(random_sample,{

    PyObject* _target, *_param;
    if (!PyArg_ParseTuple(args, "OO", &_target,&_param)) {
      return NULL;
    }

    ASSERT(PyDict_Check(_param));
    std::string dist = get_str(_param,"distribution");
    std::vector<long> shape;
    GridBase* grid = 0;
    if (PyDict_GetItemString(_param,"shape")) {
      shape = get_long_vec(_param,"shape");
      grid = get_pointer<GridBase>(_param,"grid");
    }

    if (dist == "normal") {
      std::normal_distribution<double> distribution(get_float(_param,"mu"),get_float(_param,"sigma"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid);
    } else if (dist == "cnormal") {
      cgpt_cnormal_distribution distribution(get_float(_param,"mu"),get_float(_param,"sigma"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid);
    } else if (dist == "uniform_real") {
      std::uniform_real_distribution<double> distribution(get_float(_param,"min"),get_float(_param,"max"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid);
    } else if (dist == "uniform_int") {
      std::uniform_int_distribution<int> distribution(get_int(_param,"min"),get_int(_param,"max"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid);
    } else if (dist == "zn") {
      cgpt_zn_distribution distribution(get_int(_param,"n"));
      return cgpt_random_sample(distribution,_target,cgpt_srng,cgpt_prng,shape,cgpt_seed,grid);
    } else {
      ERR("Unknown distribution: %s", dist.c_str());
    }

  });
