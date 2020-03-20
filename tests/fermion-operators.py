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
#U = g.convert(U, g.single)

# use the gauge configuration grid
grid=U[0].grid

# take slow reference implementation of wilson (kappa = 1/2/(m0 + 4) )
w_ref=g.qcd.fermion.reference.wilson(0.137,U)

# and fast Grid version
w=g.qcd.fermion.wilson_clover(U,{
    "mass" : -0.35036496350365,
    "csw_r" : 0,
    "csw_t" : 0,
    "xi_0" : 1,
    "nu" : 1,
    "isAnisotropic" : False,
    "boundary_phases" : [ 1.0, 1.0, 1.0, 1.0 ]
})

# create point source
src=g.vspincolor(grid)
src[:]=0
src[0,0,0,0]=g.vspincolor([[1,0,0],[0,0,0],[0,0,0],[0,0,0]])
src[1,0,0,0]=g.vspincolor([[1,0,0],[1,0,0],[0,0,0],[0,0,0]])

dst_ref,dst=g.lattice(src),g.lattice(src)

# correctness
w_ref.M(src,dst_ref)
w.M(src,dst)

print(g.norm2(dst-dst_ref)," / ",g.norm2(dst))

# now timing
t0 = time.time()
for i in range(100):
    w_ref.M(src,dst_ref)
t1 = time.time()
for i in range(100):
    w.M(src,dst)
t2 = time.time()

print("Reference time/s: ", t1-t0)
print("Grid time/s: ", t2-t1)



