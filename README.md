[![Build/Test](https://github.com/lehner/gpt/workflows/Build/Test/badge.svg)](https://github.com/lehner/gpt/actions?query=workflow%3ABuild%2FTest)
[![codecov](https://codecov.io/gh/lehner/gpt/branch/master/graph/badge.svg)](https://codecov.io/gh/lehner/gpt/branch/master)

![GPT Logo](/documentation/logo/logo-1280-640.png)

# GPT - Grid Python Toolkit

GPT is a [Python](https://www.python.org) measurement toolkit built on [Grid](https://github.com/paboyle/Grid) data parallelism (MPI, OpenMP, SIMD, and SIMT).
It provides a physics library for lattice QCD and related theories, a QIS module including a digital quantum computing simulator, and a machine learning module.

## System Requirements

Before installing GPT, ensure your system meets the following requirements:

- **Operating System:** Linux (Ubuntu 18.04+, Debian 10+, CentOS 7+) or macOS (10.14+)
- **CPU:** x86_64 architecture with at least AVX support (including all recent Intel and AMD CPUs), ARM architecture with NEON/SVE support (such as Apple silicon M1 and newer, A64FX)
- **GPU:** GPT can make use of accelerators using CUDA/HIP/SYCL (optional)
- **Memory:** Minimum 8GB RAM, 16GB or more recommended for larger simulations
- **Storage:** At least 10GB of free disk space
- **Python:** Version 3.6 or newer

## Prerequisites

GPT requires the following components:

1. Grid: Based on the `feature/gpt` branch of https://github.com/lehner/Grid ; please also consult [Grid's README](https://github.com/lehner/Grid/blob/feature/gpt/README.md) for supported architectures
2. Python 3.6 or newer
3. Optionally an MPI implementation (e.g., OpenMPI, MPICH); Grid needs to be compiled with "--enable-comms=none" in the absence of MPI.
4. BLAS and LAPACK libraries

## Installation

### Quick Start with Docker

The fastest way to try GPT is using Docker:

1. Install Docker on your system (https://docs.docker.com/get-docker/)
2. Run the following command to start a Jupyter notebook server:

   ```bash
   docker run --rm -p 8888:8888 gptdev/notebook
   ```

3. Open the displayed link (http://127.0.0.1:8888/?token=<token>) in your browser

Note: This session doesn't retain data after termination. To mount your current directory, use:

```bash
docker run --rm -p 8888:8888 -v $(pwd):/notebooks gptdev/notebook
```

Please consult the [GPT Docker documentation](https://github.com/lehner/gpt/tree/master/docker/README.md) for additional options.

### Local Installation

For a detailed local installation, follow these steps:

1. Clone the GPT repository:

   ```bash
   git clone https://github.com/lehner/gpt
   ```

2. Install Grid dependencies. On Ubuntu/Debian, you can use:

   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential cmake libmpich-dev libopenmpi-dev liblapack-dev libatlas-base-dev
   ```

3. Build and install Grid (here an example for a AVX2 compatible CPU):

   ```bash
   git clone https://github.com/lehner/Grid.git
   cd Grid
   git checkout feature/gpt
   ./bootstrap.sh
   mkdir build
   cd build
   ../configure --enable-simd=AVX2
   make -j
   sudo make install
   ```

4. Install GPT:

   ```bash
   cd ../gpt/lib/cgpt
   ./make
   ```

   This produces a source.sh file in the gpt/lib/cgpt/build directory.  This file needs
   to be loaded (e.g., via source gpt-path/gpt/lib/cgpt/build/source.sh) to add gpt
   to the PYTHONPATH environment variable.

   You can add a Grid build directory as an argument to the "./make" command to have serveral different builds at the same time.  This may be useful to test both CPU and GPU based builds or with and without MPI communication.

### Bootstrap Script

GPT includes bootstrap scripts for common architectures. From the GPT directory, run:

```bash
scripts/bootstrap/debian10.clang.avx2.no-mpi
```

Replace with the appropriate script for your system.

## Connecting GPT to Grid

GPT is built on top of Grid and utilizes its data parallelism features. Here's how they connect:

1. Grid provides the underlying lattice structure and parallelization.
2. GPT uses Grid's data types and parallel primitives to implement high-level physics algorithms.
3. The `feature/gpt` branch of Grid contains specific optimizations and features for GPT.

## Running GPT Efficiently

To run GPT efficiently:

1. Use MPI for distributed computing across multiple nodes.
2. Enable OpenMP for shared-memory parallelism on multi-core systems.
3. Utilize SIMD instructions (such as AVX2 or AVX-512) for vectorization.
4. For GPU acceleration, use CUDA/HIP/SYCL-enabled builds of Grid and GPT.

## Tutorials
You may also visit a static version of the tutorials [here](https://github.com/lehner/gpt/tree/master/documentation/tutorials).


## Usage

```python
import gpt as g

# Double-precision 8^4 grid
grid = g.grid([8,8,8,8], g.double)

# Parallel random number generator
rng = g.random("seed text")

# Random gauge field
U = g.qcd.gauge.random(grid, rng)

# Mobius domain-wall fermion
fermion = g.qcd.fermion.mobius(U, mass=0.1, M5=1.8, b=1.0, c=0.0, Ls=24,
                               boundary_phases=[1,1,1,-1])

# Short-cuts
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner

# Even-odd-preconditioned CG solver
slv_5d = inv.preconditioned(pc.eo2_ne(), inv.cg(eps = 1e-4, maxiter = 1000))

# Abstract fermion propagator using this solver
fermion_propagator = fermion.propagator(slv_5d)

# Create point source
src = g.mspincolor(U[0].grid)
g.create.point(src, [0, 0, 0, 0])

# Solve propagator on 12 spin-color components
prop = g( fermion_propagator * src )

# Pion correlator
g.message(g.slice(g.trace(prop * g.adj(prop)), 3))
```

