#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import sys

# load configuration
U=g.qcd.gauge.random(g.grid([8,8,8,8],g.single),g.random("test"))

# wilson, eo prec
w=g.qcd.fermion.preconditioner.eo1(g.qcd.fermion.wilson_clover(U,{
    "kappa" : 0.137,
    "csw_r" : 0,
    "csw_t" : 0,
    "xi_0" : 1,
    "nu" : 1,
    "isAnisotropic" : False,
    "boundary_phases" : [ 1.0, 1.0, 1.0, 1.0 ]
}))

# cheby
c=g.algorithms.approx.chebyshev({
    "low"   : 0.5,
    "high"  : 2.0,
    "order" : 10,
})

# implicitly restarted lanczos
irl=g.algorithms.iterative.irl({
    "Nk" : 60,
    "Nstop" : 60,
    "Nm" : 80,
    "resid" : 1e-8,
    "betastp" : 0.0,
    "maxiter" : 20,
    "Nminres" : 7,
#    "maxapply" : 100
})

# start vector
start=g.vspincolor(w.F_grid_eo)
start[:]=g.vspincolor([[1,1,1],[1,1,1],[1,1,1],[1,1,1]])

# generate eigenvectors
evec,ev=irl(c(w.NDagN), start) # , g.checkpointer("checkpoint")

# memory info
g.mem_report()

# print eigenvalues of NDagN as well
evals=g.algorithms.approx.evals(w.NDagN,evec,check_eps2=1e-11)

# deflated solver
cg=g.algorithms.iterative.cg({ "eps" : 1e-6, "maxiter" : 1000 })
defl=g.algorithms.approx.deflate(cg,evec,evals)

sol_cg = g.eval( cg(w.NDagN) * start )
eps2=g.norm2( w.NDagN * sol_cg - start ) / g.norm2(start)
niter_cg=len(cg.history)
g.message("Test resid/iter cg: ",eps2,niter_cg)
assert(eps2 < 1e-8)

sol_defl = g.eval( defl(w.NDagN) * start )
eps2=g.norm2( w.NDagN * sol_defl - start ) / g.norm2(start)
niter_defl=len(cg.history)
g.message("Test resid/iter deflated cg: ",eps2,niter_defl)
assert(eps2 < 1e-8)

assert(niter_defl < niter_cg)




