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
class cgpt_zn_distribution {
public:
  std::uniform_int_distribution<int> distribution;
  int _n;
  cgpt_zn_distribution(int n) : distribution(1,n), _n(n) {
  }

  template<typename RNG>
  ComplexD operator()(RNG& r) {
    return exp( distribution(r) * ComplexD(0.0,2.0*M_PI/(double)_n) );
  }
};

class cgpt_cnormal_distribution {
public:
  std::normal_distribution<double> distribution;
  double mu, sigma;
  cgpt_cnormal_distribution(double _mu, double _sigma) : distribution(_mu,_sigma), mu(_mu), sigma(_sigma) {
  }

  template<typename RNG>
  ComplexD operator()(RNG& r) {
    return ComplexD( distribution(r), distribution(r) );
  }
};

static void cgpt_random_to_hash(PyArrayObject* coordinates,std::vector<long>& hashes) {
  ASSERT(PyArray_NDIM(coordinates) == 2);
  long* tdim = PyArray_DIMS(coordinates);
  long nc = tdim[0];
  long nd = tdim[1];

  const int STRIDE = 512;
  const int block = 2;
  ASSERT(nd*9 < sizeof(long)*8); // make sure long is sufficient to hash

  ASSERT(PyArray_TYPE(coordinates)==NPY_INT32);
  int32_t* coor = (int32_t*)PyArray_DATA(coordinates);

  hashes.resize(nc);
  thread_for(i, nc, {
      long t = 0, s = 1;
      for (int j=0;j<nd;j++) {
	int32_t c = coor[nd*i+j] / block;
	t+=s*c;
	ASSERT(c < STRIDE);
	s*=STRIDE;
      }
      hashes[i] = t;
    });
}

template<typename T>
void cgpt_hash_offsets(std::vector<T>& h, std::map<T,std::vector<long>>& u) {
  for (long off=0;off<h.size();off++)
    u[h[off]].push_back(off);
}

template<typename T>
void cgpt_hash_unique(std::map<T,std::vector<long>>& u, std::vector<T>& h) {
  for (auto&k : u)
    h.push_back(k.first);
}

template<typename sRNG,typename pRNG>
  void cgpt_random_setup(std::vector<long>& h,sRNG & srng,std::map<long,pRNG> & prng,std::vector<long> & seed) {
  for (auto x : h) {
    auto p = prng.find(x);
    if (p == prng.end()) {
      auto & n = prng[x];
      std::vector<long> _seed = seed;
      _seed.push_back(x);
      std::seed_seq seed( _seed.begin(), _seed.end() );
      n.seed(seed);
      for (int therm = 0;therm < 10;therm++)
	n();
    }
  }
}

static void cgpt_check_global_uniqueness(std::vector<long> & unique_hashes) {
  // TODO: add this and fix block.py
  assert(0);
}

template<typename DIST,typename sRNG,typename pRNG>
  PyObject* cgpt_random_sample(DIST & dist,PyObject* _target,sRNG& srng,pRNG& prng,std::vector<long> & shape,std::vector<long> & seed) {

  if (PyArray_Check(_target)) {

    std::vector<long> hashes;
    cgpt_random_to_hash((PyArrayObject*)_target,hashes);

    std::map<long,std::vector<long>> hash_offsets;
    cgpt_hash_offsets(hashes,hash_offsets);

    std::vector<long> unique_hashes;
    cgpt_hash_unique(hash_offsets,unique_hashes);

    cgpt_check_global_uniqueness(unique_hashes);
    
    cgpt_random_setup(unique_hashes,srng,prng,seed);

    long n = 1;
    std::vector<long> dims;
    dims.push_back(hashes.size());
    for (auto s : shape) {
      n *= s;
      dims.push_back(s);
    }
    PyArrayObject* a = (PyArrayObject*)PyArray_SimpleNew((int)dims.size(), &dims[0], NPY_COMPLEX128);
    ComplexD* d = (ComplexD*)PyArray_DATA(a);
    thread_for(i, unique_hashes.size(), {
	auto h = unique_hashes[i];
	auto & uhi = hash_offsets[h];
	for (auto & x : uhi) {
	  for (long j=0;j<n;j++)
	    d[n*x+j] = dist(prng[h]);
	}
      });
    return (PyObject*)a;

  } else if (_target == Py_None) {
    ComplexD val = (ComplexD)dist(srng);
    return PyComplex_FromDoubles(val.real(),val.imag());
  } else {
    ERR("_target type not implemented");
  }

}
