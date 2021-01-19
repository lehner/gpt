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
static double fast_mul(LatticeColourMatrixD& c,
		       LatticeColourMatrixD& a,
		       LatticeColourMatrixD& b) {
  GridBase* grid = c.Grid();
  ASSERT(grid == a.Grid() &&
	 grid == b.Grid());

  autoView(c_v, c, AcceleratorWriteDiscard);
  autoView(a_v, a, AcceleratorRead);
  autoView(b_v, b, AcceleratorRead);

  auto * p_c = &c_v[0];
  auto * p_a = &a_v[0];
  auto * p_b = &b_v[0];

  double t0 = cgpt_time();
  accelerator_for(osite, grid->oSites(), grid->Nsimd(), {
      auto ma = coalescedRead(p_a[osite]);
      auto mb = coalescedRead(p_b[osite]);
      auto m = ma * mb;
      coalescedWrite(p_c[osite], m);
    });
  double t1 = cgpt_time();
  return t1-t0;
}

static void fast_mul(LatticeSpinColourMatrixD& c,
		     LatticeSpinColourMatrixD& a,
		     LatticeSpinColourMatrixD& b) {
  GridBase* grid = c.Grid();
  ASSERT(grid == a.Grid() &&
	 grid == b.Grid());
  
  autoView(c_v, c, AcceleratorWriteDiscard);
  autoView(a_v, a, AcceleratorRead);
  autoView(b_v, b, AcceleratorRead);

  auto * p_c = &c_v[0];
  auto * p_a = &a_v[0];
  auto * p_b = &b_v[0];

  accelerator_for(s, grid->oSites() * 16, grid->Nsimd(), {
      auto osite = s / 16;
      auto j = s % 4;
      auto i = (s % 16) / 4;
      auto m = coalescedRead(p_a[osite]()(i,0)) * coalescedRead(p_b[osite]()(0,j));
      for (size_t l=1;l<4;l++)
	m += coalescedRead(p_a[osite]()(i,l)) * coalescedRead(p_b[osite]()(l,j));
      coalescedWrite(p_c[osite]()(i,j), m);
    });
  
}

template<int n, typename T>
void scale_permute(T* dst, T* src, size_t outer, const Vector<T>& scale, const Vector<int>& permute, size_t inner) {

  ASSERT(permute.size() == n);
  ASSERT(scale.size() == n);
  
  auto * p = &permute[0];
  auto * s = &scale[0];

  size_t stride_o = inner * n;
  size_t stride_i = inner;
  
  accelerator_for(idx, inner * outer, 1, {
      auto i = idx % inner;
      auto o = idx / inner;
      T tmp[n];
      for (size_t l=0;l<n;l++)
	tmp[l] = src[o * stride_o + l * stride_i + i];
      for (size_t l=0;l<n;l++)
	dst[o * stride_o + l * stride_i + i] = tmp[p[l]] * s[l];
    });
}

static void fast_gamma_mul(LatticeSpinColourMatrixD& c,
		    LatticeSpinColourMatrixD& a) {
  GridBase* grid = c.Grid();
  ASSERT(grid == a.Grid());
  
  autoView(c_v, c, AcceleratorWriteDiscard);
  autoView(a_v, a, AcceleratorRead);

  auto * p_c = &c_v[0];
  auto * p_a = &a_v[0];

  // osites * [a] * 4 * 9 * Nsimd
  Vector<int> permute(4);
  Vector<ComplexD> scale(4);
  permute[0] = 3;   permute[1] = 2;   permute[2] = 1;   permute[3] = 0;
  scale[0] = ComplexD(0,1);
  scale[1] = ComplexD(0,1);
  scale[2] = ComplexD(0,-1);
  scale[3] = ComplexD(0,-1);
  scale_permute<4,ComplexD>((ComplexD*)p_c,(ComplexD*)p_a,grid->oSites(),scale,permute,4*9*grid->Nsimd());  
}

static void benchmarks(int lat) {
  std::cout << GridLogMessage << "-- Lat " << lat << std::endl;
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  Coordinate latt_size  ({lat*mpi_layout[0],lat*mpi_layout[1],lat*mpi_layout[2],lat*mpi_layout[3]});
  GridCartesian     Grid(latt_size,simd_layout,mpi_layout);

  LatticeSpinColourMatrixD a(&Grid), b(&Grid), c(&Grid), cf(&Grid);
  LatticeColourMatrixD A(&Grid), B(&Grid), C(&Grid);

  GridParallelRNG          pRNG(&Grid);      pRNG.SeedFixedIntegers(std::vector<int>({45,12,81,9}));

  double t0 = cgpt_time();
  random(pRNG,a);   random(pRNG,b);
  double t1 = cgpt_time();
  {
    double gb = 2.0 * sizeof(LatticeSpinColourMatrixD::scalar_object) * Grid._fsites / 1e9;
    std::cout << GridLogMessage << "RNG at " << gb/(t1-t0) << " GB/s" << std::endl;
  }

  random(pRNG,A);   random(pRNG,B);

  for (int i=0;i<5;i++)
    fast_mul(C,A,B);

  int N = 100;
  double gb;
  
  gb = 3.0 * sizeof(LatticeColourMatrixD::scalar_object) * Grid._fsites / 1e9 * N;
  t0 = cgpt_time();
  double dt_inner = 0.0;
  for (int i=0;i<N;i++)
    dt_inner += fast_mul(C,A,B);
  t1 = cgpt_time();
  std::cout << GridLogMessage << "SU3 " << gb << " in " << t1-t0 << " (inner " << dt_inner << ") at " << gb/(t1-t0) << " GB/s" << std::endl;

  for (int i=0;i<5;i++)
    c=a*b;    

  gb = 3.0 * sizeof(LatticeSpinColourMatrixD::scalar_object) * Grid._fsites / 1e9 * N;
  t0 = cgpt_time();
  for (int i=0;i<N;i++)
    c=a*b;
  t1 = cgpt_time();
  std::cout << GridLogMessage << gb << " in " << t1-t0 << " at " << gb/(t1-t0) << " GB/s" << std::endl;

  for (int i=0;i<5;i++)
    fast_mul(cf,a,b);
  
  t0 = cgpt_time();
  for (int i=0;i<N;i++)
    fast_mul(cf,a,b);
  t1 = cgpt_time();
  std::cout << GridLogMessage << gb << " in " << t1-t0 << " at " << gb/(t1-t0) << " GB/s" << std::endl;
  cf-=c;
  std::cout << GridLogMessage << norm2(cf) / norm2(c) << std::endl;

  gb = 2.0 * sizeof(LatticeSpinColourMatrixD::scalar_object) * Grid._fsites / 1e9 * N;
  t0 = cgpt_time();
  for (int i=0;i<N;i++)
    c = Gamma(Gamma::Algebra::GammaX) * a;
  t1 = cgpt_time();
  std::cout << GridLogMessage << "G5 application " << gb << " in " << t1-t0 << " at " << gb/(t1-t0) << " GB/s" << std::endl;

  for (int i=0;i<5;i++)
    fast_gamma_mul(cf,a);

  gb = 2.0 * sizeof(LatticeSpinColourMatrixD::scalar_object) * Grid._fsites / 1e9 * N;
  t0 = cgpt_time();
  for (int i=0;i<N;i++)
    fast_gamma_mul(cf,a);
  t1 = cgpt_time();
  std::cout << GridLogMessage << "G5 application " << gb << " in " << t1-t0 << " at " << gb/(t1-t0) << " GB/s" << std::endl;

  cf-=c;
  std::cout << GridLogMessage << norm2(cf) / norm2(c) << std::endl;

}
