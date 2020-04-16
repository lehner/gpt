#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import sys

# load configuration
U = g.load("/hpcgpfs01/work/clehner/configs/24I_0p005/ckpoint_lat.IEEE64BIG.5000")

# do everything in single-precision
U = g.convert(U, g.single)

# dwf, eo prec
w=g.qcd.fermion.preconditioner.eo2(g.qcd.fermion.zmobius(U,{
    "mass"  : 0.005,
    "M5"    : 1.8,
    "b"     : 1.0,
    "c"     : 0.0,
    "omega" : [ 1.0 ] * 16, # Mobius
    "boundary_phases" : [ 1.0, 1.0, 1.0, 1.0 ]
}))

# cheby on fine grid
c=g.algorithms.approx.chebyshev({
    "low"   : 1e-4,
    "high"  : 5.5,
    "order" : 230,
})

# implicitly restarted lanczos on fine grid
irl=g.algorithms.iterative.irl({
    "Nk" : 55,
    "Nstop" : 50,
    "Nm" : 75,
    "resid" : 1e-8,
    "betastp" : 0.0,
    "maxiter" : 20,
    "Nminres" : 0,
#    "maxapply" : 100
})

# start vector
start=g.vspincolor(w.F_grid_eo)
start[:]=g.vspincolor([[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
start.checkerboard(g.odd) 
# generate odd-site eigenvalues to be consistent
# with old code
try:
    evec,ev_cheb=g.load("/hpcgpfs01/scratch/clehner/basis", { "grids" : w.F_grid_eo })
except NotImplementedError:
    # generate eigenvectors
    evec,ev_cheb=irl(c(w.NDagN), start, g.checkpointer("/hpcgpfs01/scratch/clehner/checkpointOdd"))
    g.save("/hpcgpfs01/scratch/clehner/basis",(evec,ev))

ev_basis=g.algorithms.approx.evals(w.NDagN,evec,check_eps2 = 1e-8)

# memory info
g.meminfo()

# cheby on coarse grid
c=g.algorithms.approx.chebyshev({
    "low"   : 4.2e-4,
    "high"  : 5.5,
    "order" : 150,
})

# implicitly restarted lanczos on coarse grid
irl=g.algorithms.iterative.irl({
    "Nk" : 160,
    "Nstop" : 150,
    "Nm" : 180,
    "resid" : 1e-8,
    "betastp" : 0.0,
    "maxiter" : 20,
    "Nminres" : 0,
#    "maxapply" : 100
})

# coarse-grid
grid_coarse=g.block.grid(w.F_grid_eo,[16,2,2,2,2])
start=g.vcomplex(grid_coarse,50)
start[:]=g.vcomplex([ 1 ] * 50,50)

# basis
basis=evec
g.message("Ortho round 1")
g.block.orthogonalize(grid_coarse,basis)
g.message("Ortho round 2")
g.block.orthogonalize(grid_coarse,basis)
g.message("Ortho round 3")
g.block.orthogonalize(grid_coarse,basis)

# now define coarse-grid operator
cop=g.block.operator(c(w.NDagN),grid_coarse,basis)

tmp=g.lattice(start)
cop(start,tmp)

#g.message(tmp[0,0,0,0])

tmpf=g.lattice(basis[0])
g.block.promote(tmp,tmpf,basis)
g.block.project(start,tmpf,basis)

g.message(g.norm2(start-tmp))

sys.exit(0)

coarse_evec,coarse_ev=irl(cop, start, g.checkpointer("/hpcgpfs01/scratch/clehner/checkpoint3"))

sys.exit(0)

# print eigenvalues of NDagN as well
for i,v in enumerate(evec):
    w.NDagN(v,start)
    l=g.innerProduct(v,start).real
    g.message("%d %g %g %g" % (i,l,ev[i],c(l)))
