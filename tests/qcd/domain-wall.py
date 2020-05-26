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
#U = g.load("/hpcgpfs01/work/clehner/configs/16I_0p01_0p04/ckpoint_lat.IEEE64BIG.1100")
U=g.qcd.gauge.random(g.grid([8,8,8,8],g.double),g.random("test"),scale = 0.5)
g.message("Plaquette:",g.qcd.gauge.plaquette(U))

# do everything in single-precision
#U = g.convert(U, g.single)

# use the gauge configuration grid
grid=U[0].grid

# mobius <> zmobius domain wall quark
qm=g.qcd.fermion.mobius(U,{
    "mass"  : 0.04,
    "M5"    : 1.8,
    "b"     : 1.5,
    "c"     : 0.5,
    "Ls" : 24,
    "boundary_phases" : [ 1.0, 1.0, 1.0, 1.0 ]
})

w=g.qcd.fermion.wilson_clover(U,{
    "mass" : -1.8,
    "csw_r" : 0,
    "csw_t" : 0,
    "xi_0" : 1,
    "nu" : 1,
    "isAnisotropic" : False,
    "boundary_phases" : [ 1.0, 1.0, 1.0, 1.0 ]
})

def H(s,d):
    w.M(s,d)
    g.eval(d,g.gamma[5]*d / 2.0)

# find largest eigenvalue of overlap kernel
pi=g.algorithms.iterative.power_iteration({ "eps" : 1e-3, "maxiter" : 30 })
st=g.vspincolor(grid)
g.random("test").cnormal(st)
pi(H,st)

qz=g.qcd.fermion.zmobius(U,{
    "mass"  : 0.04,
    "M5"    : 1.8,
    "b"     : 1.0,
    "c"     : 0.0,
    "omega" : [
1.8530255403075104 + 1j*(0),

1.593983619964445 + 1j*(0),

1.2127961700946597 + 1j*(0),

0.8619627395089003 + 1j*(0),

0.5906659797999617 + 1j*(0),

0.39706365411914263 + 1j*(0),

0.26344003875987015 + 1j*(0),

0.17207275433484948 + 1j*(0),

0.11348176576169644 + 1j*(0.01959818142922749),

0.11348176576169644 + 1j*(-0.01959818142922749),

0.07479396293100343 + 1j*(0.07180640088469024),

0.07479396293100343 + 1j*(-0.07180640088469024),
    ],
    "boundary_phases" : [ 1.0, 1.0, 1.0, 1.0 ]
})

# create point source
src=g.mspincolor(grid)
g.create.point(src, [0,0,0,0])

# solver
s=g.qcd.fermion.solver
cg=g.algorithms.iterative.cg({
    "eps" : 1e-8,
    "maxiter" : 1000
})
slv_qm=s.propagator(s.eo_ne(g.qcd.fermion.preconditioner.eo2(qm), cg))
slv_qz=s.propagator(s.eo_ne(g.qcd.fermion.preconditioner.eo2(qz), cg))

# propagator
dst_qm=g.mspincolor(grid)
dst_qz=g.mspincolor(grid)
slv_qm(src,dst_qm)
slv_qz(src,dst_qz)

# two-point
correlator_qm=g.slice(g.trace(dst_qm*g.adj(dst_qm)),3)
correlator_qz=g.slice(g.trace(dst_qz*g.adj(dst_qz)),3)
correlator_ref=[
    0.5805132985115051,
    0.2003495693206787,
    0.05248865857720375,
    0.02542622946202755,
    0.015156489796936512,
    0.009473282843828201,
    0.005946052260696888,
    0.003997922874987125,
    0.0027736490592360497,
    0.001767154666595161,
    0.0010931078577414155,
    0.0007353568216785789,
    0.0004973784671165049,
    0.0003497783036436885,
    0.0002569338248576969,
]

# output
eps_qm=0.0
eps_qz=0.0
for t in range(len(correlator_qm)):
    eps_qm+=abs(correlator_qm[t].real - correlator_ref[t])
    eps_qz+=abs(correlator_qz[t].real - correlator_ref[t])
    g.message(t,correlator_qm[t].real,correlator_qz[t].real,correlator_ref[t])
g.message("Test results: %g %g" % (eps_qm,eps_qz))
assert(eps_qm < 1e-7)
assert(eps_qz < 1e-3)
g.message("Test successful")


