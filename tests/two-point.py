#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

# load configuration
U = g.load("/hpcgpfs01/work/clehner/configs/16I_0p01_0p04/ckpoint_lat.IEEE64BIG.1100")

# do everything in single-precision
U = g.convert(U, g.single)

# use the gauge configuration grid
grid=U[0].grid

# take slow reference implementation of wilson
w=g.qcd.fermion.reference.wilson(0.137,U)

# create point source
src=g.mspincolor(grid)
g.create.point(src, [0,0,0,0])

# build solver using g5m
slv=g.qcd.fermion.solver.g5m(1e-6,1000)

# propagator
dst=g.mspincolor(grid)
slv(w,dst,src)

# two-point
correlator=g.slice(g.trace(dst*g.adj(dst)),3)

# output
for t,c in enumerate(correlator):
    print(t,c.real)

