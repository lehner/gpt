#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g

grid_dp=g.grid([4,4,4,4],g.double)
grid_sp=g.grid([4,4,4,4],g.single)

for grid,eps in [ (grid_dp,1e-15), (grid_sp,1e-7) ]:
    rng=g.random("test")
    m=g.mcolor(grid)
    rng.lie(m)
    m2=g.matrix.exp(g.matrix.log(m))
    eps2=g.norm2(m-m2) / g.norm2(m)
    g.message(eps2)

