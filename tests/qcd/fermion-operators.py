#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys, cmath

# load configuration
rng=g.random("test")
U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], g.double), rng)

# do everything in single-precision
U = g.convert(U, g.single)

# plaquette
g.message("Plaquette:",g.qcd.gauge.plaquette(U))

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
    "boundary_phases" : [ cmath.exp(1j), cmath.exp(2j), cmath.exp(3j), cmath.exp(4j) ]
}

#pf=g.params("~/gpt/tests/wilson.txt")
#print(pf)

# take slow reference implementation of wilson (kappa = 1/2/(m0 + 4) ) ;
w_ref=g.qcd.fermion.reference.wilson(U,p)

# and fast Grid version
w=g.qcd.fermion.wilson_clover(U,p,kappa = 0.137)

# create point source
src=rng.cnormal(g.vspincolor(grid))

dst_ref,dst=g.lattice(src),g.lattice(src)

# correctness
dst_ref @= w_ref.M * src
dst @= w.M * src

eps=g.norm2(dst-dst_ref)/g.norm2(dst)
g.message("Test wilson versus reference:", eps)
assert(eps<1e-13)

# now timing
t0 = g.time()
for i in range(100):
    w_ref.M(dst_ref,src)
t1 = g.time()
for i in range(100):
    w.M(dst,src)
t2 = g.time()
for i in range(100):
    dst = w.M(src)
t3 = g.time()
for i in range(100):
    dst @= w.M * src
t4 = g.time()

g.message("Reference time/s: ", t1-t0)
g.message("Grid time/s (reuse lattices): ", t2-t1)
g.message("Grid time/s (with temporaries): ", t3-t2)
g.message("Grid time/s (with expressions): ", t4-t3)

# create point source
src=g.mspincolor(grid)
g.create.point(src, [1,0,0,0]) # pick point 1 so that "S" in preconditioner contributes to test

# build solver using g5m and cg
s=g.qcd.fermion.solver
cg=g.algorithms.iterative.cg({
    "eps" : 1e-6,
    "maxiter" : 1000
})

slv=s.propagator(s.inv_g5m_ne(w, cg))
slv_eo1=s.propagator(s.inv_eo_ne(g.qcd.fermion.preconditioner.eo1(w), cg))
slv_eo2=s.propagator(s.inv_eo_ne(g.qcd.fermion.preconditioner.eo2(w), cg))

# propagator
dst_eo1=g.mspincolor(grid)
dst_eo2=g.mspincolor(grid)

dst_eo1 @= slv_eo1 * src
iter_eo1=len(cg.history)

dst_eo2 @= slv_eo2 * src
iter_eo2=len(cg.history)

eps2=g.norm2(dst_eo1 - dst_eo2) / g.norm2(dst_eo1)
g.message(f"Result of test EO1 versus EO2 preconditioning: eps2={eps2} iter1={iter_eo1} iter2={iter_eo2}")
assert(eps2 < 1e-12)

# true residuum
eps2=g.norm2(w.M * dst_eo1 - src) / g.norm2(src)
g.message("Result of M M^-1 = 1 test: eps2=",eps2)
assert(eps2 < 1e-10)

# and a reference
if True:
    dst=g.mspincolor(grid)
    dst @= slv * src
    eps2=g.norm2(dst_eo1 - dst) / g.norm2(dst_eo1)
    g.message("Result of test EO1 versus G5M: eps2=",eps2)
    assert(eps2 < 1e-10)

dst=dst_eo2

# two-point
correlator=g.slice(g.trace(dst*g.adj(dst)),3)

# test value of correlator
correlator_ref=[ 
    1.0648243427276611,
    0.09166733920574188,
    0.01625053584575653,
    0.003848351538181305,
    0.0011231594253331423,
    0.0003689819131977856,
    0.00013113138265907764,
    5.2903757023159415e-05,
    3.4834869438782334e-05,
    5.4076463129604235e-05,
    0.00013475494051817805,
    0.00037725106813013554,
    0.00115416687913239,
    0.003970596939325333,
    0.01652490347623825,
    0.09108280390501022
]

# output
for t,c in enumerate(correlator):
    g.message(t,c.real,correlator_ref[t])

eps=np.linalg.norm(np.array(correlator) - np.array(correlator_ref))
g.message("Expected correlator eps: ", eps)
assert(eps<1e-5)
