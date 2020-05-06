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

# output
for t in range(len(correlator_qm)):
    g.message(t,correlator_qm[t].real,correlator_qz[t].real)
