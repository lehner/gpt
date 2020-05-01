#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Production code to generate coarse-grid eigenvectors using existing
# fine-grid basis vectors
#
import gpt as g
import sys
import numpy as np

# parameters
fn=g.default.get("--params","params.txt")
params=g.params(fn,verbose = True)

# load configuration
U = params["config"]

# fermion
q=params["fmatrix"](U)

# load basis vectors
fg_basis,fg_cevec,fg_feval = g.load(params["basis"],{
    "grids" : q.F_grid_eo
})

# memory info
g.meminfo()

# prepare and test basis
basis=[]
nbasis=params["nbasis"]
for i in range(nbasis):
    basis.append( g.vspincolor(q.F_grid_eo) )
    g.block.promote(fg_cevec[i],basis[i],fg_basis)
    g.algorithms.approx.evals(q.NDagN,[ basis[i] ],check_eps2=1e-4)
    g.message("Compare to: %g" % fg_feval[i])

# now discard original basis
del fg_basis
del fg_cevec
g.message("Memory information after discarding original basis:")
g.meminfo()

# coarse grid
cgrid=params["cgrid"](q.F_grid_eo)

# cheby on coarse grid
cop=params["cmatrix"](q.NDagN,cgrid,basis)

# implicitly restarted lanczos on coarse grid
irl=params["method_evec"]

# start vector
cstart=g.vcomplex(cgrid,nbasis)
cstart[:]=g.vcomplex([ 1 ] * nbasis,nbasis)

# basis
northo=params["northo"]
for i in range(northo):
    g.message("Orthonormalization round %d" % i)
    g.block.orthonormalize(cgrid,basis)

# now define coarse-grid operator
ftmp=g.lattice(basis[0])
ctmp=g.lattice(cstart)
g.block.promote(cstart,ftmp,basis)
g.block.project(ctmp,ftmp,basis)
g.message("Test precision of promote-project chain: %g" % (g.norm2(cstart-ctmp)/g.norm2(cstart)))

try:
    cevec,cev=g.load("cevec", { "grids" : cgrid })
except g.LoadError:
    cevec,cev=irl(cop, cstart, params["checkpointer"])
    g.save("cevec",(cevec,cev))

# smoother
smoother=params["smoother"]
nsmoother=params["nsmoother"]
v_fine=g.lattice(basis[0])
v_fine_smooth=g.lattice(basis[0])
try:
    ev3=g.load("ev3")
except g.LoadError:
    ev3=[ 0.0 ] * len(cevec)
    for i,v in enumerate(cevec):
        g.block.promote(v,v_fine,basis)
        for j in range(nsmoother):
            v_fine_smooth[:]=0
            smoother(q.NDagN,v_fine,v_fine_smooth)
            v_fine @= v_fine_smooth / g.norm2(v_fine_smooth)**0.5
        ev_smooth=g.algorithms.approx.evals(q.NDagN,[ v_fine ],check_eps2 = 1e-2)
        ev3[i] = ev_smooth[0]
        g.message("Eigenvalue %d = %.15g" % (i,ev3[i]))
    g.save("ev3",ev3)

# tests
start=g.lattice(basis[0])
start[:]=g.vspincolor([[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
start *= 1.0 / g.norm2(start)**0.5

def save_history(fn,history):
    f=open(fn,"wt")
    for i,v in enumerate(history):
        f.write("%d %.15E\n" % (i,v))
    f.close()

solver=g.algorithms.approx.coarse_deflate(params["test_solver"], cevec,basis,ev3)
v_fine[:]=0
solver(q.NDagN,start,v_fine)
save_history("cg_test.defl_all_ev3",solver.inverter.history)

solver=g.algorithms.approx.coarse_deflate(params["test_solver"], cevec[0:len(basis)],basis,ev3[0:len(basis)])
v_fine[:]=0
solver(q.NDagN,start,v_fine)
save_history("cg_test.defl_full",solver.inverter.history)

solver=params["test_solver"]
v_fine[:]=0
solver(q.NDagN,start,v_fine)
save_history("cg_test.undefl",solver.history)

# save in rbc format
g.save("lanczos.output", [basis,cevec,ev3], params["format"])

