#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import sys

# test SYM1
U = g.load("/hpcgpfs01/work/clehner/configs/96I/test/ckpoint_lat.2000")
U = g.convert(U, g.single)

q=g.qcd.fermion.preconditioner.eo1(g.qcd.fermion.mobius(U,{
    "mass"  : 0.01,
    "M5"    : 1.8,
    "b"     : 1.5,
    "c"     : 0.5,
    "Ls"    : 12,
    "boundary_phases" : [ 1.0, 1.0, 1.0, -1.0 ]
}))

# load configuration
basis,cevec,feval = g.load("/hpcgpfs01/work/clehner/configs/96I/test/checkpoint", { #lanczos.output
    "grids" : q.F_grid_eo
})

# test fine evec of basis
tmp=g.vspincolor(q.F_grid_eo)
for i in range(4):
    g.block.promote(cevec[i],tmp,basis)
    g.algorithms.approx.evals(q.NDagN,[ tmp ],check_eps2=1e-5)
    g.message(feval[i])

sys.exit(0)

c=g.algorithms.approx.chebyshev({
    "low"   : 0.01,
    "high"  : 6.25,
    "order" : 50,
})

cop=g.block.operator(c(q.NDagN),cevec[0].grid,basis)

ev_basis=g.algorithms.approx.evals(cop,cevec,check_eps2 = 1000.0,skip = 10)

sys.exit(0)


# test SYM2
U = g.load("/hpcgpfs01/work/clehner/configs/16I_0p01_0p04/ckpoint_lat.IEEE64BIG.1100")
U = g.convert(U, g.single)

q=g.qcd.fermion.preconditioner.eo2(g.qcd.fermion.zmobius(U,{
    "mass"  : 0.00107,
    "M5"    : 1.8,
    "b"     : 1.0,
    "c"     : 0.0,
    "omega" : [
        1.0903256131299373,
        0.9570283702230611,
        0.7048886040934104,
        0.48979921782791747,
        0.328608311201356,
        0.21664245377015995,
        0.14121112711957107,
        0.0907785101745156,
        0.05608303440064219 -0.007537158177840385j,
        0.05608303440064219 +0.007537158177840385j,
        0.0365221637144842 -0.03343945161367745j,
        0.0365221637144842 +0.03343945161367745j
    ],
    "boundary_phases" : [ 1.0, 1.0, 1.0, -1.0 ]
}))

# load configuration
basis,cevec,feval = g.load("/hpcgpfs01/work/clehner/runs/tune-lanc-16c/job-00004/lanczos.output", {
    "grids" : q.F_grid_eo
})

c=g.algorithms.approx.chebyshev({
    "low"   : 0.004,
    "high"  : 5.5,
    "order" : 30,
})

cop=g.block.operator(c(q.NDagN),cevec[0].grid,basis)

ev_basis=g.algorithms.approx.evals(cop,cevec,check_eps2 = 1e-1)

