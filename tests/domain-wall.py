#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys
import time

# load configuration
U = g.load("/hpcgpfs01/work/clehner/configs/16I_0p01_0p04/ckpoint_lat.IEEE64BIG.1100")

# do everything in single-precision
U = g.convert(U, g.single)

# use the gauge configuration grid
grid=U[0].grid

# zmobius domain wall quark
q=g.qcd.fermion.mobius(U,{
    "mass"  : 0.01,
    "M5"    : 1.8,
    "b"     : 1.5, #1.0
    "c"     : 0.5, #0.0
    #"omega" : [ 0.5 ] * 12, # Mobius
    "Ls" : 12,
    "boundary_phases" : [ 1.0, 1.0, 1.0, 1.0 ]
})

# create point source
src=g.mspincolor(grid)
g.create.point(src, [0,0,0,0])

# solver
s=g.qcd.fermion.solver
cg=g.algorithms.iterative.cg({
    "eps" : 1e-6,
    "maxiter" : 1000
})
slv=s.propagator(s.eo_ne(g.qcd.fermion.preconditioner.eo2(q), cg))

# propagator
dst=g.mspincolor(grid)
slv(src,dst)

# two-point
correlator=g.slice(g.trace(dst*g.adj(dst)),3)

# output
for t,c in enumerate(correlator):
    g.message(t,c.real)
