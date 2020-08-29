[![Build/Test](https://github.com/lehner/gpt/workflows/Build/Test/badge.svg)](https://github.com/lehner/gpt/actions?query=workflow%3ABuild%2FTest)
[![codecov](https://codecov.io/gh/lehner/gpt/branch/master/graph/badge.svg)](https://codecov.io/gh/lehner/gpt/branch/master)

![GPT Logo](/documentation/logo/logo-1280-640.png)

# GPT - Grid Python Toolkit

GPT is a [Python](https://www.python.org) measurement toolkit built on [Grid](https://github.com/paboyle/Grid) data parallelism.  It provides a physics library for lattice QCD and related theories.

## Installation
GPT is developed with the feature/gpt branch of https://github.com/lehner/Grid.

## Setting up the runtime
### Docker
An easy way to quickly test `gpt`, is to use the [prebuild Docker images](https://github.com/lehner/gpt/tree/master/docker).

First make sure you have [Docker](https://docs.docker.com/get-docker/) installed.

To start a Jupyter notebook server with the latest gpt version, execute:
```
docker run --rm -p 8888:8888 gptdev/notebook
```

the server is then bound to the local host system on port `8888`.

To run, for example the [gpt tutorials](https://github.com/lehner/gpt#tutorials), use `-v` to mount the directory into the Docker image, i.e.
```
docker run --rm -v /local/path/to/gpt-repository/documentation/tutorial:/notebooks/tutorial -p 8888:8888 gptdev/notebook
```

Open the shown link `http://127.0.0.1:8888/?token=<token>`, in a browser. You should then see the `tutorial` folder, from within you can run the interactive tutorials.

If you prefer an interactive python3 console run
```
docker run --rm -it -p 8888:8888 gptdev/play /usr/bin/python3
```

or to obtain a bash terminal
```
docker run --rm -it -p 8888:8888 gptdev/play /bin/bash
```

Use `-v /local/gpt/code:/gpt-code` to mount your custom code into the Docker container.

For more information refer to the [Docker docs](https://docs.docker.com/get-started/).

### Manual
```bash
source gpt/scripts/source.sh
```

## Tutorials
A good starting point to learn how to use GPT is the [Tutorials Section](https://github.com/lehner/gpt/tree/master/documentation/tutorial)
with interactive [Jupyter](https://jupyter.org/) notebooks.

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
