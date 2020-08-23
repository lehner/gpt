/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2020  Daniel Richtmann (daniel.richtmann@ur.de)

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

#include <Grid/Grid.h>
#include "../lib/cgpt/lib/lib.h"

using namespace Grid;

#ifndef NBASIS
#define NBASIS 40
#endif

std::vector<int> readFromCommandlineIvec(int*                    argc,
                                         char***                 argv,
                                         std::string&&           option,
                                         const std::vector<int>& defaultValue) {
  std::string      arg;
  std::vector<int> ret(defaultValue);
  if(GridCmdOptionExists(*argv, *argv + *argc, option)) {
    arg = GridCmdOptionPayload(*argv, *argv + *argc, option);
    GridCmdOptionIntVector(arg, ret);
  }
  return ret;
}

int readFromCommandlineInt(int* argc, char*** argv, const std::string& option, int defaultValue) {
  std::string arg;
  int         ret = defaultValue;
  if(GridCmdOptionExists(*argv, *argv + *argc, option)) {
    arg = GridCmdOptionPayload(*argv, *argv + *argc, option);
    GridCmdOptionInt(arg, ret);
  }
  return ret;
}

template<typename vCoeff_t>
void run_benchmark(int* argc, char*** argv) {
  static_assert(getPrecision<vCoeff_t>::value == 2 || getPrecision<vCoeff_t>::value == 1,
                "Incorrect precision"); // double or single

  const int nbasis = NBASIS;
  static_assert((nbasis & 0x1) == 0, "");

  GridCartesian* fgrid =
    SpaceTimeGrid::makeFourDimGrid(readFromCommandlineIvec(argc, argv, "--fgrid", {16, 16, 16, 32}),
                                   GridDefaultSimd(Nd, vComplex::Nsimd()),
                                   GridDefaultMpi());
  GridCartesian* cgrid =
    SpaceTimeGrid::makeFourDimGrid(readFromCommandlineIvec(argc, argv, "--cgrid", {4, 4, 4, 8}),
                                   GridDefaultSimd(Nd, vComplex::Nsimd()),
                                   GridDefaultMpi());

  int N = readFromCommandlineInt(argc, argv, "--N", 1000);

  std::cout << GridLogMessage << "Lookup Table Benchmark with" << std::endl;
  std::cout << GridLogMessage << "    fine fdimensions    : " << fgrid->_fdimensions << std::endl;
  std::cout << GridLogMessage << "    coarse fdimensions  : " << cgrid->_fdimensions << std::endl;
  std::cout << GridLogMessage << "    precision           : " << (getPrecision<vCoeff_t>::value == 2 ? "double" : "single") << std::endl;
  std::cout << GridLogMessage << "    nbasis              : " << nbasis << std::endl << std::endl;

  typedef Lattice<iSpinColourVector<vCoeff_t>>                        FineVector;
  typedef Lattice<typename FineVector::vector_object::tensor_reduced> FineComplex;
  typedef Lattice<iVector<iSinglet<vCoeff_t>, nbasis>>                CoarseVector;

  // Source and destination
  FineVector   src(fgrid);
  CoarseVector dst_default(cgrid);
  CoarseVector dst_lut(cgrid);
  CoarseVector dst_nolut(cgrid);

  // Basis
  std::vector<FineVector> basis(nbasis, fgrid);

  // Randomize
  GridParallelRNG rng(fgrid);
  rng.SeedFixedIntegers({1, 2, 3, 4});
  gaussian(rng, src);
  for(auto& b : basis) gaussian(rng, b);

  // Lookup table
  FineComplex mask_full(fgrid);
  mask_full = 1.;
  cgpt_lookup_table<FineComplex> lut_full(cgrid, mask_full);

  // Flops
  double flops_per_site = 1.0 * (4 * Nc * 6 + (4 * Nc - 1) * 2) * nbasis;
  double flops          = flops_per_site * src.Grid()->gSites() * N;
  double prec_bytes     = getPrecision<vCoeff_t>::value * 4;
  double nbytes =
    ((nbasis * 2 * 4 * Nc + 2 * 4 * Nc) * src.Grid()->gSites() + 2 * nbasis * dst_default.Grid()->gSites()) *
    prec_bytes * N;

  // Warmup default
  for(auto n : {1, 2, 3, 4, 5}) vectorizableBlockProject(dst_default, src, basis);

  // Time default
  double t0 = usecond();
  for(int n = 0; n < N; n++) vectorizableBlockProject(dst_default, src, basis);
  double t1 = usecond();

  // Warmup with lookup table
  for(auto n : {1, 2, 3, 4, 5}) vectorizableBlockProjectUsingLut(dst_lut, src, basis, lut_full);

  // Time with lookup table
  double t2 = usecond();
  for(int n = 0; n < N; n++) vectorizableBlockProjectUsingLut(dst_lut, src, basis, lut_full);
  double t3 = usecond();

  // Warmup without lookup table
  for(auto n : {1, 2, 3, 4, 5}) vectorizableBlockProjectUsingNoLut(dst_nolut, src, basis);

  // Time without lookup table
  double t4 = usecond();
  for(int n = 0; n < N; n++) vectorizableBlockProjectUsingNoLut(dst_nolut, src, basis);
  double t5 = usecond();

  // Report default
  double dt           = (t1 - t0) / 1e6;
  double GFlopsPerSec = flops / dt / 1e9;
  double GBPerSec     = nbytes / dt / 1e9;
  std::cout << GridLogMessage << N << " applications of vectorizableBlockProject" << std::endl;
  std::cout << GridLogMessage << "    Time to complete            : " << dt << " s" << std::endl;
  std::cout << GridLogMessage << "    Total performance           : " << GFlopsPerSec << " GFlops/s" << std::endl;
  std::cout << GridLogMessage << "    Effective memory bandwidth  : " << GBPerSec << " GB/s" << std::endl << std::endl;

  // Report with lookup table
  dt           = (t3 - t2) / 1e6;
  GFlopsPerSec = flops / dt / 1e9;
  GBPerSec     = nbytes / dt / 1e9;
  std::cout << GridLogMessage << N << " applications of vectorizableBlockProjectUsingLut" << std::endl;
  std::cout << GridLogMessage << "    Time to complete            : " << dt << " s" << std::endl;
  std::cout << GridLogMessage << "    Total performance           : " << GFlopsPerSec << " GFlops/s" << std::endl;
  std::cout << GridLogMessage << "    Effective memory bandwidth  : " << GBPerSec << " GB/s" << std::endl << std::endl;

  // Report without lookup table
  dt           = (t5 - t4) / 1e6;
  GFlopsPerSec = flops / dt / 1e9;
  GBPerSec     = nbytes / dt / 1e9;
  std::cout << GridLogMessage << N << " applications of vectorizableBlockProjectUsingNoLut" << std::endl;
  std::cout << GridLogMessage << "    Time to complete            : " << dt << " s" << std::endl;
  std::cout << GridLogMessage << "    Total performance           : " << GFlopsPerSec << " GFlops/s" << std::endl;
  std::cout << GridLogMessage << "    Effective memory bandwidth  : " << GBPerSec << " GB/s" << std::endl << std::endl;
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  run_benchmark<vComplexF>(&argc, &argv);
  run_benchmark<vComplexD>(&argc, &argv);

  Grid_finalize();
}
