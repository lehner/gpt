[![Build/Test](https://github.com/lehner/gpt/workflows/Build/Test/badge.svg)](https://github.com/lehner/gpt/actions?query=workflow%3ABuild%2FTest)
[![codecov](https://codecov.io/gh/lehner/gpt/branch/master/graph/badge.svg)](https://codecov.io/gh/lehner/gpt/branch/master)

![GPT Logo](/documentation/logo/logo-1280-640.png)

# GPT - Grid Python Toolkit

GPT is a [Python](https://www.python.org) measurement toolkit built on [Grid](https://github.com/paboyle/Grid) data parallelism (MPI, OpenMP, SIMD, and SIMT).
It provides a physics library for lattice QCD and related theories, a QIS module including a digital quantum computing simulator, and a machine learning module.

## System Requirements

Before installing GPT, ensure your system meets the following requirements:

- **Operating System:** Linux (Ubuntu 18.04+, Debian 10+, CentOS 7+) or macOS (10.14+)
- **CPU:** x86_64 architecture with AVX2 support (Intel Haswell or newer, AMD Excavator or newer)
- **Memory:** Minimum 8GB RAM, 16GB or more recommended for larger simulations
- **Storage:** At least 10GB of free disk space
- **Python:** Version 3.6 or newer

## Prerequisites

GPT requires the following components:

1. Grid: Based on the `feature/gpt` branch of https://github.com/lehner/Grid
2. Python 3.6 or newer
3. MPI implementation (e.g., OpenMPI, MPICH)
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

### Local Installation

For a detailed local installation, follow these steps:

1. Clone the GPT repository:

   ```bash
   git clone https://github.com/lehner/gpt
   cd gpt
   ```

2. Install Grid dependencies. On Ubuntu/Debian, you can use:

   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential cmake libmpich-dev libopenmpi-dev liblapack-dev libatlas-base-dev
   ```

3. Build and install Grid:

   ```bash
   git clone https://github.com/lehner/Grid.git
   cd Grid
   git checkout feature/gpt
   ./bootstrap.sh
   mkdir build
   cd build
   ../configure --enable-simd=AVX2
   make -j$(nproc)
   sudo make install
   ```

4. Install GPT:

   ```bash
   cd ../../
   pip install -e .
   ```

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
3. Utilize SIMD instructions (AVX2 or AVX-512) for vectorization.
4. For GPU acceleration, use CUDA-enabled builds of Grid and GPT.

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
