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

# mobius <> zmobius domain wall quark
qm=g.qcd.fermion.mobius(U,{
    "mass"  : 0.04,
    "M5"    : 1.8,
    "b"     : 1.5,
    "c"     : 0.5,
    "Ls" : 24,
    "boundary_phases" : [ 1.0, 1.0, 1.0, 1.0 ]
})

qz=g.qcd.fermion.zmobius(U,{
    "mass"  : 0.04,
    "M5"    : 1.8,
    "b"     : 1.0,
    "c"     : 0.0,
    "omega" : [
        1.45806438985048,
        1.18231318389348,
        0.830951166685955,
        0.542352409156791,
        0.341985020453729,
        0.21137902619029,
        0.126074299502912,
        0.0990136651962626,
        0.0686324988446592 + 0.0550658530827402j,
        0.0686324988446592 - 0.0550658530827402j
    ],
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
    0.00018832855857908726,
    0.00016109236457850784,
    0.00017106340965256095,
    0.00020944568677805364,
    0.00029460625955834985,
    0.00045832418254576623,
    0.000751025159843266,
    0.001192546333186328,
    0.001964164897799492,
    0.003390846075490117,
    0.005769025068730116,
    0.008645612746477127,
    0.0126266498118639,
    0.0183967035263776,
    0.0275730341672897,
    0.0572328567504882,
    0.2176910340785980
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


