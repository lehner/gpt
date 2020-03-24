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

# wilson parameters
p={
    "kappa" : 0.137,
    "csw_r" : 0,
    "csw_t" : 0,
    "xi_0" : 1,
    "nu" : 1,
    "isAnisotropic" : False,
    "boundary_phases" : [ 1.0, 1.0, 1.0, 1.0 ]
}

#pf=g.params("~/gpt/tests/wilson.txt")
#print(pf)

# take slow reference implementation of wilson (kappa = 1/2/(m0 + 4) ) ; 
w_ref=g.qcd.fermion.reference.wilson(U,p)

# and fast Grid version
w=g.qcd.fermion.wilson_clover(U,p)

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

# create point source
src=g.mspincolor(grid)
g.create.point(src, [0,0,0,0])

# build solver using g5m and cg
s=g.qcd.fermion.solver
cg=g.algorithms.iterative.cg({
    "eps" : 1e-6,
    "maxiter" : 1000
})
#slv=s.propagator(s.g5m_ne(w, cg))

slv=s.propagator(s.eo_ne(g.qcd.fermion.preconditioner.eo2(w), cg))

# propagator
dst=g.mspincolor(grid)
slv(src,dst)

# two-point
correlator=g.slice(g.trace(dst*g.adj(dst)),3)

# output
for t,c in enumerate(correlator):
    print(t,c.real)
