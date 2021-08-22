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


template<typename vobj_a, typename vobj_b>
void mk_binary_mul_ll(const micro_kernel_arg_t & arg, size_t i0, size_t i1, size_t n_subblock) {
  typedef decltype(vobj_a()*vobj_b()) vobj_c;
  typedef typename vobj_c::scalar_object sobj_c;
  
  micro_kernel_view(vobj_a, a_p, 0);
  micro_kernel_view(vobj_b, b_p, 1);
  micro_kernel_view(vobj_c, c_p, 2);

  micro_kernel_for(idx, i1-i0, sizeof(vobj_c)/sizeof(sobj_c), n_subblock, {
      coalescedWrite(a_p[idx], coalescedRead(b_p[idx]) * coalescedRead(c_p[idx]));
    });
}

  
template<typename Lat>
void micro_kernels(int lat) {
  
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  Coordinate latt_size  ({lat*mpi_layout[0],lat*mpi_layout[1],lat*mpi_layout[2],lat*mpi_layout[3]});
  GridCartesian     Grid(latt_size,simd_layout,mpi_layout);

  typedef typename Lat::vector_object vobj;
  typedef typename Lat::scalar_object sobj;
  Lat a(&Grid), b(&Grid), c(&Grid), d(&Grid);

  std::cout << GridLogMessage << lat << "^4" << std::endl;
    
  GridParallelRNG          pRNG(&Grid);      pRNG.SeedFixedIntegers(std::vector<int>({45,12,81,9}));
  random(pRNG,a);   random(pRNG,b);

  int Nwarm = 10;
  int N = 500;
  double gb, t0, t1, t2, t3, t4, t5, t0b, t1b;
  mk_timer t_et, t_eti;
  std::map<std::string, mk_timer> t_mk;
  std::vector<micro_kernel_blocking_t> blockings = {
#ifdef GRID_HAS_ACCELERATOR
    { 8*1024, 1 },
    { 32*1024, 1 },
    { 256*1024, 1 },
#else
    { 512, 8 },
    { 512, 16 },
    { 512, 32 },
    { 512, 64 },
    { 256, 8 },
    { 256, 16 },
    { 256, 32 },
    { 128, 8 },
    { 128, 16 }
#endif
  };
 
  gb = 4.0 * 3.0 * sizeof(sobj) * Grid._fsites / 1e9;
  for (int i=0;i<Nwarm+N;i++) {
    t0 = cgpt_time();
    c = a*b;
    d = a*c;
    c = a*b;
    d = a*c;
    t1 = cgpt_time();
    if (i>=Nwarm)
      t_et.add(t1-t0);
  }

  Lat d_copy = a*a*b;

  for (int i=0;i<Nwarm+N;i++) {
    t0 = cgpt_time();
    d = a*a*b;
    d = a*a*b;
    t1 = cgpt_time();
    if (i>=Nwarm)
      t_eti.add(t1-t0);
  }

  d = Zero();
  
  t2 = cgpt_time();
  std::vector<micro_kernel_t> expression;
  micro_kernel_arg_t views_c_a_b, views_d_a_c;

  views_c_a_b.add(c, AcceleratorWriteDiscard, false);
  views_c_a_b.add(a, AcceleratorRead);
  views_c_a_b.add(b, AcceleratorRead);

  views_d_a_c.add(d, AcceleratorWriteDiscard);
  views_d_a_c.add(a, AcceleratorRead);
  views_d_a_c.add(c, AcceleratorRead, false);

  // TODO: internal index size
  expression.push_back({ mk_binary_mul_ll<vobj,vobj>, views_c_a_b });
  expression.push_back({ mk_binary_mul_ll<vobj,vobj>, views_d_a_c });
  expression.push_back({ mk_binary_mul_ll<vobj,vobj>, views_c_a_b });
  expression.push_back({ mk_binary_mul_ll<vobj,vobj>, views_d_a_c });

  t3 = cgpt_time();

  for (auto b : blockings) {
    mk_timer t;
    for (int i=0;i<Nwarm+N;i++) {
      t0 = cgpt_time();
      eval_micro_kernels(expression, b);
      t1 = cgpt_time();
      if (i>=Nwarm)
        t.add(t1-t0);
    }
    char buf[256];
    sprintf(buf,"MK %d-%d",b.block_size,b.subblock_size);
    t_mk[buf] = t;
  }
  t5 = cgpt_time();

  views_c_a_b.release();
  views_d_a_c.release();

  d -= d_copy;
  double err2 = norm2(d);

  t_et.print ("GridET separate", gb);
  t_eti.print("GridET joint   ", gb);
  for (auto t : t_mk)
    t.second.print (t.first, gb);
  
}

template<typename Lat>
void mk_bench_mul() {
  micro_kernels<Lat>(4);
  micro_kernels<Lat>(6);
  micro_kernels<Lat>(8);
  micro_kernels<Lat>(10);
  micro_kernels<Lat>(12);
  micro_kernels<Lat>(16);
#ifdef GRID_HAS_ACCELERATOR
  micro_kernels<Lat>(24);
  micro_kernels<Lat>(32);
  micro_kernels<Lat>(48);
#endif
}

EXPORT(benchmarks,{
    //mask();
    //half();
    //benchmarks(8);
    //benchmarks(16);
    //benchmarks(32);
    std::cout << GridLogMessage << std::endl << std::endl << "Benchmarking ComplexD" << std::endl << std::endl;
    mk_bench_mul<LatticeComplexD>();

    std::cout << GridLogMessage << std::endl << std::endl << "Benchmarking ColourD" << std::endl << std::endl;
    mk_bench_mul<LatticeColourMatrixD>();

    //std::cout << GridLogMessage << std::endl << std::endl << "Benchmarking SpinColourD" << std::endl << std::endl;
    //mk_bench_mul<LatticeSpinColourMatrixD>();

    return PyLong_FromLong(0);
  });

EXPORT(tests,{
    test_global_memory_system();
    return PyLong_FromLong(0);
  });
