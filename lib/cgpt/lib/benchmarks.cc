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

/*
template<typename uncompressed_t, typename compressed_t>
class iCompressed {
public:
  typedef typename uncompressed_t::scalar_type scalar_type;
  typedef typename uncompressed_t::scalar_object scalar_object;
  typedef typename uncompressed_t::vector_type vector_type;
  typedef typename uncompressed_t::vector_typeD vector_typeD;
  typedef iCompressed<uncompressed_t,compressed_t> this_t;

  compressed_t data;

  accelerator iCompressed() = default;

  iCompressed(const Zero & zz) {
  }

  friend accelerator_inline scalar_object Reduce(const this_t & a) {
    return scalar_object();
  }
  
  friend accelerator_inline this_t operator+(const this_t & a, const this_t & b) {
    return this_t();
  }

  friend accelerator_inline uncompressed_t coalescedRead(const this_t& c) {
    return uncompressed_t();
  }

  friend accelerator_inline void coalescedWrite(this_t& c, const uncompressed_t & u) {
  }

  friend accelerator_inline auto innerProductD(const this_t& left,
					       const this_t& right)
    -> decltype(innerProductD(uncompressed_t(),uncompressed_t())) {
    return innerProductD(uncompressed_t(),uncompressed_t());
  }

  friend void cgpt_lattice_convert_from(Lattice<this_t>& dst,cgpt_Lattice_base* src) {
  }

  friend int singlet_rank(const this_t& c) {
    return singlet_rank(uncompressed_t());
  }

  friend const std::string get_otype(const Lattice<this_t>& l) {
    return "";
  }

};

template<typename uncompressed_t, typename compressed_t>
class GridTypeMapper<iCompressed<uncompressed_t,compressed_t>> {
public:
  constexpr static int count = GridTypeMapper<uncompressed_t>::count;
};

void mask() {

    int lat = 8;
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  Coordinate latt_size  ({lat*mpi_layout[0],lat*mpi_layout[1],lat*mpi_layout[2],lat*mpi_layout[3]});
  GridCartesian     Grid(latt_size,simd_layout,mpi_layout);

  //cgpt_Lattice< iCompressed< iSinglet<vComplexD>, short > > a(&Grid);

  //autoView(a_v,a,CpuWrite);
  //std::cout << GridLogMessage << "Test " << sizeof(a_v[0]) << std::endl;

  //GridParallelRNG          pRNG(&Grid);      pRNG.SeedFixedIntegers(std::vector<int>({45,12,81,9}));
  //random(pRNG,b);
  //a = Zero();
}
*/

EXPORT(benchmarks,{
    //mask();
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
