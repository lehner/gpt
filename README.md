# GPT - Grid production tools

GPT is a Python3 measurement interface for [Grid](https://github.com/paboyle/Grid).

## Installation
```bash
source gpt/scripts/source.sh
```

## Usage

```python
import gpt as g

# load gauge field and describe fermion
gauge=g.qcd.gaugefield("params.txt")
light=g.qcd.fermion("light.txt")

# create point source
src=g.qcd.propagator(gauge.dp.grid)
g.create.point(src, [0,0,0,0])

# solve
prop=light.solve.exact(src)

# pion
corr_pion=g.slice(g.adj(prop)*prop,3)
print("Pion two point:")
print(corr_pion)

# vector
gamma=g.qcd.gamma
corr_vector=g.slice(gamma[0]*g.adj(prop)*gamma[0]*prop,3)
print("Vector two point:")
print(corr_vector)
```
