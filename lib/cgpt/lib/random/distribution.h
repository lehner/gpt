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
template<typename RNG,typename T>
class cgpt_random {
protected:
  RNG rng;
  T state;
  int nbits;

  std::vector<RealD> stack_normal;

  static constexpr T log2(T n) {
    return (n > 1) ? 1 + log2(n >> 1) : 0;
  }

public:
  cgpt_random(const std::vector<uint64_t> & seed) : rng(seed), state(0), nbits(0) {
    populate();
  }

  void populate() {
    state = state * rng.size() + rng();
    nbits += rng.l2size();
    if (nbits > sizeof(T) * 8)
      nbits=sizeof(T) * 8;
  }

  T get_bits(int bits) {
    // get bits is slow
    while (bits > nbits)
      populate();

    T base = ((T)1) << bits;
    T res = state & (base - 1);
    state /= base;
    nbits-=bits;
    return res;
  }

  uint32_t get_uint32_t() {
    return get_bits(32);
  }

  RealD get_double() { // uniform in [0,1[
    return (RealD)get_bits(53) / (RealD)( ((T)1) << 53 );
  }

  T get_uniform_int(T max) { // uniform in {0,1,...,max}

    if (max == 0)
      return 0;

    int bits = (int)log2(max)+1;
    do {
      T res = get_bits(bits);
      if (res <= max)
	return res;
    } while(true);

    return 0;
  }

  T get_uniform_int(T min, T max) { // uniform in {min,...,max}
    return get_uniform_int(max - min) + min;
  }

  RealD get_normal() {
    static const RealD epsilon = std::numeric_limits<RealD>::min();
    static const RealD two_pi = 2.0*3.14159265358979323846;
    
    long n_stack = stack_normal.size();
    if (n_stack > 0) {
      RealD ret = stack_normal[n_stack - 1];
      stack_normal.resize(n_stack - 1);
      return ret;
    }

    // Box-Muller
    RealD u1, u2;
    do {
      u1 = get_double();
      u2 = get_double();
    } while (u1 <= epsilon);

    RealD z0, z1;
    z0 = ::sqrt(-2.0 * ::log(u1)) * ::cos(two_pi * u2);
    z1 = ::sqrt(-2.0 * ::log(u1)) * ::sin(two_pi * u2);
    stack_normal.push_back(z1);
    return z0;      
  }

  RealD get_normal(RealD mu, RealD sigma) {
    return get_normal() * sigma + mu;
  }

  ComplexD get_zn(int n) {
    return exp( ((ComplexD)get_uniform_int(n-1)) * ComplexD(0.0,2.0*M_PI/(RealD)n) );
  }

};

typedef cgpt_random< cgpt_vrng_ranlux24_389_64, uint64_t > cgpt_random_vectorized_ranlux24_389_64;
typedef cgpt_random< cgpt_vrng_ranlux24_24_64, uint64_t > cgpt_random_vectorized_ranlux24_24_64;

// distribution interface
class cgpt_normal_distribution {
 public:
  RealD mu,sigma;
  cgpt_normal_distribution(RealD _mu, RealD _sigma) : mu(_mu), sigma(_sigma) {};
  template<typename R> RealD operator()(R & r) { return r.get_normal(mu,sigma); };
};

class cgpt_cnormal_distribution {
 public:
  RealD mu,sigma;
  cgpt_cnormal_distribution(RealD _mu, RealD _sigma) : mu(_mu), sigma(_sigma) {};
  template<typename R> ComplexD operator()(R & r) {
    RealD im = r.get_normal(mu,sigma);
    RealD re = r.get_normal(mu,sigma);
    return ComplexD(re,im);
  };
};

class cgpt_uniform_real_distribution {
 public:
  RealD min,max;
  cgpt_uniform_real_distribution(RealD _min, RealD _max) : min(_min), max(_max) {};
  template<typename R> RealD operator()(R & r) { return r.get_double() * (max-min) + min; };
};

class cgpt_uniform_int_distribution {
 public:
  int min,max;
  cgpt_uniform_int_distribution(int _min, int _max) : min(_min), max(_max) {};
  template<typename R> int operator()(R & r) { return r.get_uniform_int(max - min) + min; };
};

class cgpt_zn_distribution {
 public:
  int n;
  cgpt_zn_distribution(int _n) : n(_n) {};
  template<typename R> ComplexD operator()(R & r) { return r.get_zn(n); };
};
