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
template<typename base_vrng,typename base_rng,int max_vlen>
class cgpt_vector_rng {
public:
  typedef typename base_rng::base_type int_type;
  typedef typename base_rng::container_type int_ctype;
  typedef typename base_vrng::base_type vint_type;
  typedef typename base_vrng::container_type vint_ctype;
  static const long vlen = sizeof(vint_type) / sizeof(int_type);
  static const long nvrng= max_vlen / vlen + ((max_vlen % vlen == 0) ? 0 : 1);

protected:

  base_rng rng;
  base_vrng vrng[nvrng];

  std::vector<int_type> buffer;
  long nbuffer;

  // first seed rng using sha256(s)
  int_ctype seed(const std::vector<uint64_t> & s) {
    assert(rng.seed_size * sizeof(int_type) % sizeof(uint32_t) == 0);
    long nwords = rng.seed_size * sizeof(int_type) / sizeof(uint32_t);
    int_ctype r(nwords);
    std::vector<uint32_t> rw;
    long idx = 0;
    while (rw.size() < nwords) {

      std::vector<uint64_t> tmp = s;
      tmp.push_back(idx++);

      uint32_t sha256_seed[8];
      cgpt_sha256(sha256_seed,&tmp[0],sizeof(uint64_t) * tmp.size());

      for (int w=0;w<8;w++)
	rw.push_back(sha256_seed[w]);

    }
    memcpy(&r[0],&rw[0],sizeof(int_type) * nwords);
    return r;
  }

  // then seed a large number of rngs
  // fill one lane after another, so that strategy is independent of vlen !
  vint_ctype vseed() {
    //std::cout << GridLogMessage << "vlen = " << vlen << " vlen_max = " << max_vlen << " nvrng= " << nvrng << std::endl;
    vint_ctype r(base_vrng::seed_size);
    int_type* p = (int_type*)&r[0];
    for (long lane=0;lane<vlen;lane++) {
      for (long j=0;j<base_vrng::seed_size;j++)
	rng(p[vlen*j + lane]);
    }
    //std::cout << GridLogMessage << "vState = " << r << std::endl;
    return r;
  }

public:

  cgpt_vector_rng(const std::vector<uint64_t>& _seed) : buffer(vlen * nvrng) {
    rng.seed(seed(_seed));
    for (long i=0;i<nvrng;i++)
      vrng[i].seed(vseed());
    populate();
  }

  void populate() {
    nbuffer=0;
    long j = 0;
    vint_ctype dst(1);
    vint_type & r = dst[0];
    for (long i=0;i<nvrng;i++) {
      vrng[i](r);
      for (long lane=0;lane<vlen;lane++)
	buffer[j++] = ((int_type*)&r)[lane];
    }      
    //std::cout << GridLogMessage << "Buffer: " << buffer << std::endl;
  }

  int_type operator()() {
    if (nbuffer == max_vlen) // only use up to max_vlen to keep numbers independent of vlen/architecture
      populate();

    return buffer[nbuffer++];
  }

  long size() {
    return vrng[0].size();
  }

  int l2size() {
    return vrng[0].l2size();
  }

};

typedef cgpt_vector_rng< cgpt_base_vrng_ranlux24_389, cgpt_base_rng_ranlux24_389, 64 > cgpt_vrng_ranlux24_389_64;
typedef cgpt_vector_rng< cgpt_base_vrng_ranlux24_24, cgpt_base_rng_ranlux24_24, 64 > cgpt_vrng_ranlux24_24_64;
