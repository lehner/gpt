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


    Works for Integer and vInteger.

    
    For details, see http://luscher.web.cern.ch/luscher/ranlux/notes.pdf (*) .

*/

template<typename C,int l2b,int r,int s,int p>
class cgpt_ranlux {
protected:
  long b;
  C x, temporaries;
  int offset, discard;

#define _X_(i) x[(i-1 - offset + r) % r]
#define c_nm1 temporaries[0]
#define vbm1 temporaries[1]
#define vb temporaries[2]
#define Deltan temporaries[3]

public:

  typedef typename std::remove_reference<decltype(x[0])>::type base_type;
  typedef C container_type;
  static const int seed_size = r;

  // constructor seeds
  cgpt_ranlux() : x(r), temporaries(5),
    offset(0), b(1 << l2b), discard(r-1) {

    assert(s >= 1 && s < r);
    assert(sizeof(long)*8 >= l2b);
    // make sure type is unsigned
    assert( (Integer)(-1) > (Integer)(0) );
    // precompute
    vbm1 = b-1;
    vb   = b; 
  }

  void seed(const container_type & seed) {
    assert(seed.size() == r);
    // avoid trivial sequence, see (5) and (6) of (*)
    for (int i=0;i<r;i++)
      x[i] = seed[i] & vbm1;
    c_nm1 = seed[0] == 0;
  }

  void write_state() {
    std::cout << GridLogMessage << "cgpt_ranlux::write_state()" << std::endl;
    for (long i=1;i<=r;i++)
      std::cout << GridLogMessage << "x_{n-" << i << "} = " << _X_(i) << std::endl;
    std::cout << GridLogMessage << "c_{n-1} = " << c_nm1 << std::endl;
  }

  void step(base_type & dst) {
    offset = (offset + 1) % r;
    Deltan = _X_(s) - _X_(r) - c_nm1;
    //c_nm1 = (Deltan < 0); // for signed integers
    c_nm1 = (Deltan >= vb); // for unsigned integers
    Deltan += vb;
    _X_(0) = Deltan & vbm1;
    dst = _X_(0);
  }

  void operator()(base_type & dst) {
    if (++discard == r) {
      discard=0;
      for (int i=0;i<p-r;i++)
        step(dst);
    }
    step(dst);
  }

  long size() {
    return b;
  }
  
  int l2size() {
    return l2b;
  }

#undef _X_
#undef c_nm1
#undef vmb1
#undef vb
#undef Deltan
};

// slow, ideal generator with full decorrelation
typedef cgpt_ranlux<ThreadSafeAlignedVector<vInteger>,24,24,10,389> cgpt_base_vrng_ranlux24_389;
typedef cgpt_ranlux<std::vector<Integer>,24,24,10,389> cgpt_base_rng_ranlux24_389;

// fast, imperfect generator for testing
typedef cgpt_ranlux<ThreadSafeAlignedVector<vInteger>,24,24,10,24> cgpt_base_vrng_ranlux24_24;
typedef cgpt_ranlux<std::vector<Integer>,24,24,10,24> cgpt_base_rng_ranlux24_24;
