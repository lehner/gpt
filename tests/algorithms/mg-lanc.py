#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import sys

# load configuration
U = g.load("/hpcgpfs01/work/clehner/configs/96I/test/ckpoint_lat.2000")

# do everything in single-precision
U = g.convert(U, g.single)

# dwf, eo prec
w=g.qcd.fermion.preconditioner.eo1(g.qcd.fermion.mobius(U,{
    "mass"  : 0.01,
    "M5"    : 1.8,
    "b"     : 1.5,
    "c"     : 0.5,
    "Ls"    : 12,
    "boundary_phases" : [ 1.0, 1.0, 1.0, -1.0 ]
}))

# cheby on fine grid
c=g.algorithms.approx.chebyshev({
    "low"   : 0.01,
    "high"  : 6.25,
    "order" : 50,
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
except g.LoadError:
    # generate eigenvectors
    evec,ev_cheb=irl(c(w.NDagN), start, g.checkpointer("/hpcgpfs01/scratch/clehner/checkpointOdd"))
    g.save("/hpcgpfs01/scratch/clehner/basis",(evec,ev_cheb))

ev_basis=g.algorithms.approx.evals(w.NDagN,evec,check_eps2 = 1e-8)

# memory info
g.mem_report()

# cheby on coarse grid
c=g.algorithms.approx.chebyshev({
    "low"   : 0.01,
    "high"  : 6.25,
    "order" : 50,
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
grid_coarse=g.block.grid(w.F_grid_eo,[12,2,2,2,2])
cstart=g.vcomplex(grid_coarse,50)
cstart[:]=g.vcomplex([ 1 ] * 50,50)

# basis
basis=evec
g.message("Ortho round 1")
g.block.orthonormalize(grid_coarse,basis)
g.message("Ortho round 2")
g.block.orthonormalize(grid_coarse,basis)

# now define coarse-grid operator
cop=g.block.operator(c(w.NDagN),grid_coarse,basis)

tmp=g.lattice(cstart)
cop(cstart,tmp)

#g.message(tmp[0,0,0,0])

tmpf=g.lattice(basis[0])
g.block.promote(tmp,tmpf,basis)
g.block.project(cstart,tmpf,basis)

g.message(g.norm2(cstart-tmp)/g.norm2(cstart))

try:
    coarse_evec,coarse_ev=g.load("/hpcgpfs01/scratch/clehner/coarse_basis", { "grids" : grid_coarse })
except g.LoadError:
    # generate eigenvectors
    coarse_evec,coarse_ev=irl(cop, cstart, g.checkpointer("/hpcgpfs01/scratch/clehner/checkpoint4"))
    g.save("/hpcgpfs01/scratch/clehner/coarse_basis",(coarse_evec,coarse_ev))

# smoother
smoother=g.algorithms.iterative.cg({
    "eps": 1e-8,
    "maxiter": 10
})
n_smoother_iter=3

v_fine=g.lattice(evec[0])
v_fine_smooth=g.lattice(evec[0])
try:
    ev3=g.load("/hpcgpfs01/scratch/clehner/ev3")
except g.LoadError:
    ev3=[ 0.0 ] * len(coarse_evec)
    for i,v in enumerate(coarse_evec):
        g.block.promote(v,v_fine,basis)
        for j in range(n_smoother_iter):
            v_fine_smooth[:]=0
            smoother(w.NDagN,v_fine,v_fine_smooth)
            v_fine @= v_fine_smooth / g.norm2(v_fine_smooth)**0.5
        ev_smooth=g.algorithms.approx.evals(w.NDagN,[ v_fine ],check_eps2 = 1e-5)
        ev3[i] = ev_smooth[0]
        g.message("Eigenvalue %d = %.8g" % (i,ev3[i]))
    g.save("/hpcgpfs01/scratch/clehner/ev3",ev3)


# CG tests
start *= 1.0 / g.norm2(start)**0.5
solver=g.algorithms.approx.coarse_deflate(
    g.algorithms.iterative.cg({
        "eps" : 1e-8,
        "maxiter" : 1000
    }),
    coarse_evec,basis,ev3)
v_fine[:]=0
solver(w.NDagN,start,v_fine)
cg_defl_full_ev3=solver.inverter.history

solver=g.algorithms.approx.coarse_deflate(
    g.algorithms.iterative.cg({
        "eps" : 1e-8,
        "maxiter" : 1000
    }),
    coarse_evec[0:len(basis)],basis,ev3[0:len(basis)])
v_fine[:]=0
solver(w.NDagN,start,v_fine)
cg_defl_basis=solver.inverter.history

solver=g.algorithms.iterative.cg({
    "eps" : 1e-8,
    "maxiter" : 1000
})
v_fine[:]=0
solver(w.NDagN,start,v_fine)
cg_undefl=solver.history
