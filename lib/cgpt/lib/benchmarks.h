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


    This file provides a playground for benchmarking new C++ functions
    before they go into production.

*/

#include "expression/mul.h"

static void benchmarks(int lat) {
  std::cout << GridLogMessage << "-- Lat " << lat << std::endl;
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  Coordinate latt_size  ({lat*mpi_layout[0],lat*mpi_layout[1],lat*mpi_layout[2],lat*mpi_layout[3]});
  GridCartesian     Grid(latt_size,simd_layout,mpi_layout);

  LatticeSpinColourMatrixD a(&Grid), b(&Grid), c(&Grid);

  GridParallelRNG          pRNG(&Grid);      pRNG.SeedFixedIntegers(std::vector<int>({45,12,81,9}));

  double t0 = cgpt_time();
  random(pRNG,a);   random(pRNG,b);
  double t1 = cgpt_time();
  {
    double gb = 2.0 * sizeof(LatticeSpinColourMatrixD::scalar_object) * Grid._fsites / 1e9;
    std::cout << GridLogMessage << "RNG at " << gb/(t1-t0) << " GB/s" << std::endl;
  }

  int N = 100;
  double gb;

  gb = 3.0 * sizeof(LatticeSpinColourMatrixD::scalar_object) * Grid._fsites / 1e9 * N;
  cgpt_Lattice_base* dst = 0;
  for (int i=0;i<=N;i++) {
    if (i==1)
      t0 = cgpt_time();
    dst = lattice_mul(dst,false,0,a,0,b,0,1.0);
  }
  t1 = cgpt_time();
  std::cout << GridLogMessage << gb << " in " << t1-t0 << " at " << gb/(t1-t0) << " GB/s" << std::endl;

  
  gb = 3.0 * sizeof(LatticeSpinColourMatrixD::scalar_object) * Grid._fsites / 1e9 * N;
  t0 = cgpt_time();
  for (int i=0;i<N;i++) {
    c = a*b;
  }
  t1 = cgpt_time();
  std::cout << GridLogMessage << gb << " in " << t1-t0 << " at " << gb/(t1-t0) << " GB/s" << std::endl;


}
