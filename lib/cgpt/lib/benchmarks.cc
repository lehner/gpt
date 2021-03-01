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
#include "benchmarks.h"

#if 0
template<typename base_t, typename int_t>
class iHalfPrecision {
public:
  typedef typename base_t::scalar_object base_sobj_t;
  typedef typename base_t::scalar_type base_Coeff_t;
  static constexpr int nFloats = 2 * sizeof(base_sobj_t) / sizeof(base_Coeff_t);
  static constexpr int nSimd = sizeof(base_t) / sizeof(base_sobj_t);

  typedef iHalfPrecision< base_sobj_t, int_t > scalar_object;
  typedef void scalar_type; // if we need this we are in trouble
  typedef void vector_type; // or maybe we should just inherit from base?

  int_t exponents[nSimd];
  int_t mantissa[nSimd][nFloats];

  iHalfPrecision<base_t,int_t> operator=(const Zero & zero) {
    // TODO
    return *this;
  }
};

void half() {

  int lat = 8;
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  Coordinate latt_size  ({lat*mpi_layout[0],lat*mpi_layout[1],lat*mpi_layout[2],lat*mpi_layout[3]});
  GridCartesian     Grid(latt_size,simd_layout,mpi_layout);

  iHalfPrecision< vSpinColourVectorF, int16_t > hp;
  std::cout << GridLogMessage << hp.nFloats << ", " << hp.nSimd << " , " << sizeof(hp) << " versus " << sizeof(vSpinColourVectorF) << std::endl;
  
  //Lattice< iVector<vComplexF,3> > b(&Grid);
  Lattice< iHalfPrecision< vSpinColourVectorF, int16_t > > a(&Grid), b(&Grid);

  GridParallelRNG          pRNG(&Grid);      pRNG.SeedFixedIntegers(std::vector<int>({45,12,81,9}));
  //random(pRNG,b);

  a = Zero();
  //b = a + a;
}
#endif

EXPORT(benchmarks,{
    //half();
    benchmarks(8);
    benchmarks(16);
    benchmarks(32);
    return PyLong_FromLong(0);
  });

EXPORT(tests,{
    test_global_memory_system();
    return PyLong_FromLong(0);
  });
