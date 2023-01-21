[![Build/Test](https://github.com/lehner/gpt/workflows/Build/Test/badge.svg)](https://github.com/lehner/gpt/actions?query=workflow%3ABuild%2FTest)
[![codecov](https://codecov.io/gh/lehner/gpt/branch/master/graph/badge.svg)](https://codecov.io/gh/lehner/gpt/branch/master)

![GPT Logo](/documentation/logo/logo-1280-640.png)

# GPT - Grid Python Toolkit

GPT is a [Python](https://www.python.org) measurement toolkit built on [Grid](https://github.com/paboyle/Grid) data parallelism (MPI, OpenMP, SIMD, and SIMT).
It provides a physics library for lattice QCD and related theories, a QIS module including a digital quantum computing simulator, and a machine learning module.

## Quick Start
The fastest way to try GPT is to install [Docker](https://docs.docker.com/get-docker/),
start a [Jupyter](https://jupyter.org/) notebook server with the latest GPT version by running
```
docker run --rm -p 8888:8888 gptdev/notebook
```
and then open the shown link `http://127.0.0.1:8888/?token=<token>` in a browser.
You should see the tutorials folder pre-installed.

Note that this session does not retain data after termination.  Run
```
docker run --rm -p 8888:8888 -v $(pwd):/notebooks gptdev/notebook
```
to instead mount the current working directory on your machine.

Please consult the [GPT Docker documentation](https://github.com/lehner/gpt/tree/master/docker/README.md) for additional options.


## Installation
A detailed description on how to install GPT
locally can be found [here](README.setup.md).


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
