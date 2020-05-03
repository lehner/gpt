#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import sys
import numpy as np

# load configuration
#/hpcgpfs01/work/clehner/runs/tune-lanc-48c/job-strange-01000/lanczos.output
#basis,cevec,feval = g.load("/hpcgpfs01/work/lqcd/k2pipipbc/chulwoo/32ID/0.0001/evecs/job-900/lanczos.output",{
#    "grids" : g.grid([12,32,32,32,64],g.single,g.redblack)
#})

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

# save in different layout
g.save("/hpcgpfs01/work/clehner/configs/96I/test/checkpoint2",
       [basis,cevec,feval], 
       g.format.cevec({
           "nsingle" : len(basis) // 2,
           "max_read_blocks" : 16,
           "mpi" : [ 1, 2,2,2,2 ]
       }))

# and load again to verify
basis2,cevec2,feval2 = g.load("/hpcgpfs01/work/clehner/configs/96I/test/checkpoint2", {
    "grids" : q.F_grid_eo
})

assert(len(basis) == len(basis2))
assert(len(cevec) == len(cevec2))
assert(len(feval) == len(feval2))

pos=g.coordinates(basis[i])
eps=0.0
for i in range(len(basis)):
    A=basis[i][pos]
    B=basis2[i][pos]
    eps+=q.F_grid_eo.globalsum(float(np.linalg.norm(A-B)))
g.message("Test basis: %g" % (eps))

pos=g.coordinates(cevec[i])
eps=0.0
for i in range(len(cevec)):
    A=cevec[i][pos]
    B=cevec2[i][pos]
    eps+=q.F_grid_eo.globalsum(float(np.linalg.norm(A-B)))
g.message("Test cevec: %g" % (eps))

eps=0.0
for i in range(len(feval)):
    eps+=(feval[i] - feval2[i])**2.0
g.message("Test eval: %g" % (eps))

sys.exit(0)

# test eigenvectors
c=g.algorithms.approx.chebyshev({
    "low"   : 0.01,
    "high"  : 6.25,
    "order" : 50,
})

cop=g.block.operator(c(q.NDagN),cevec[0].grid,basis)

ev_basis=g.algorithms.approx.evals(cop,cevec,check_eps2 = 1000.0,skip = 10)


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

