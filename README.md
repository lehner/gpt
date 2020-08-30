[![Build/Test](https://github.com/lehner/gpt/workflows/Build/Test/badge.svg)](https://github.com/lehner/gpt/actions?query=workflow%3ABuild%2FTest)
[![codecov](https://codecov.io/gh/lehner/gpt/branch/master/graph/badge.svg)](https://codecov.io/gh/lehner/gpt/branch/master)

![GPT Logo](/documentation/logo/logo-1280-640.png)

# GPT - Grid Python Toolkit

GPT is a [Python](https://www.python.org) measurement toolkit built on [Grid](https://github.com/paboyle/Grid) data parallelism.  It provides a physics library for lattice QCD and related theories.

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

# load gauge field and describe fermion
gauge=g.qcd.gaugefield("params.txt")
light=g.qcd.fermion("light.txt")

# create point source
src=g.mspincolor(gauge.dp.grid)
g.create.point(src, [0,0,0,0])

# solve
prop=light.solve.exact(src)

# pion
corr_pion=g.slice(g.trace(g.adj(prop)*prop),3)
print("Pion two point:")
print(corr_pion)

# vector
gamma=g.gamma
corr_vector=g.slice(g.trace(gamma[0]*g.adj(prop)*gamma[0]*prop),3)
print("Vector two point:")
print(corr_vector)
```
