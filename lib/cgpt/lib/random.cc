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
#include "random/mem.h"
#include "random/ranlux.h"
#include "random/vector.h"
#include "random/distribution.h"
#include "random/parallel.h"
#include "random/engine.h"

/*
  rng[hash[pos]]

  hash[pos] : divide lattice into 2^nd blocks/hashes
              need to make sure that no two ranks have same hash

	      for 32^4 local volume, this means 65536 local hashes

	      when do we send all hashes do root to check for duplicates
*/

EXPORT(create_random,{

#if 0
    {
      std::vector<long> seed = { 1,2,3 };
      cgpt_vrng_ranlux24_794_64 vtest(seed);
      cgpt_rng_ranlux24_794_64 stest(seed);
      
      for (int i=0;i<1024*100;i++) {
	long a = vtest();
	long b = stest();
	assert(a == b);
      }
      
      double t0 = cgpt_time();
      for (int i=0;i<1024*100;i++) {
	long a = vtest();
      }
      double t1 = cgpt_time();
      for (int i=0;i<1024*100;i++) {
	long a = stest();
      }
      double t2 = cgpt_time();
      std::cout << GridLogMessage << "Timing: " << (t1-t0) << " and " << (t2-t1) << std::endl;
      
      cgpt_random_vectorized_ranlux24_794_64 rnd(seed);
      std::cout << GridLogMessage << rnd.get_normal() << std::endl;
    }
#endif



    PyObject* _type,* _seed;
    std::string type, seed;
    if (!PyArg_ParseTuple(args, "OO", &_type,&_seed)) {
      return NULL;
    }
    cgpt_convert(_type,type);
    cgpt_convert(_seed,seed);

    if (type == "vectorized_ranlux24_389_64") {
      return PyLong_FromVoidPtr(new cgpt_random_engine< cgpt_random_vectorized_ranlux24_389_64 >(seed));
    } else if (type == "vectorized_ranlux24_24_64") {
      return PyLong_FromVoidPtr(new cgpt_random_engine< cgpt_random_vectorized_ranlux24_24_64 >(seed));
    } else {
      ERR("Unknown rng engine type %s",type.c_str());
    }

    return PyLong_FromLong(0);
    
  });

EXPORT(delete_random,{

    void* _p;
    if (!PyArg_ParseTuple(args,"l", &_p)) {
      return NULL;
    }

    cgpt_random_engine_base* p = (cgpt_random_engine_base*)_p;
    delete p;

    return PyLong_FromLong(0);
  });

EXPORT(random_sample,{

    PyObject* _param;
    void* _p;
    if (!PyArg_ParseTuple(args, "lO", &_p,&_param)) {
      return NULL;
    }

    cgpt_random_engine_base* p = (cgpt_random_engine_base*)_p;
    return p->sample(_param);

  });

// the following allow the bigcrush test to link directly against this
struct cgpt_rng_test {
  std::vector<cgpt_random_engine_base*> engines;
  std::vector<uint32_t> buf_bits;
  std::vector<double> buf_double;
};

void* cgpt_rng_test_create(int iengine) {
  if (iengine == 0) {
    cgpt_rng_test* p = new cgpt_rng_test();
    p->engines.resize(64);
    thread_for(i,p->engines.size(), {
	char buf[256];
	sprintf(buf,"big crush test %d",(int)i);
	(p->engines)[i] = new cgpt_random_engine< cgpt_random_vectorized_ranlux24_389_64 >(buf);
      });
    return (void*)p;
  }
  return 0;
}

void cgpt_rng_test_destroy(void* t) {
  cgpt_rng_test* p = (cgpt_rng_test*)t;
  for (auto x : p->engines)
    delete x;
  delete p;
}

double cgpt_rng_test_GetU01(void* param, void* state) {
  cgpt_rng_test* p = (cgpt_rng_test*)state;
  long n = p->buf_double.size();
  if (n == 0) {
    p->buf_double.resize(p->engines.size());
    thread_for(i,p->engines.size(), {
	p->buf_double[i]=p->engines[i]->test_U01();
      });
    n=p->buf_double.size();
  }

  double r=p->buf_double[n-1];
  p->buf_double.resize(n-1);
  return r;
}

unsigned long cgpt_rng_test_GetBits(void* param, void* state) {
  cgpt_rng_test* p = (cgpt_rng_test*)state;
  long n = p->buf_bits.size();
  if (n == 0) {
    p->buf_bits.resize(p->engines.size());
    thread_for(i,p->engines.size(), {
	p->buf_bits[i]=p->engines[i]->test_bits();
      });
    n=p->buf_bits.size();
  }

  unsigned long r=p->buf_bits[n-1];
  p->buf_bits.resize(n-1);
  return r;
}


void cgpt_rng_test_Write(void* state) {
}




