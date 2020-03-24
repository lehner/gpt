#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import sys

# load configuration
U = g.load("/hpcgpfs01/work/clehner/configs/16I_0p01_0p04/ckpoint_lat.IEEE64BIG.1100")

# do everything in single-precision
U = g.convert(U, g.single)

# wilson, eo prec
w=g.qcd.fermion.preconditioner.eo2(g.qcd.fermion.wilson_clover(U,{
    "kappa" : 0.137,
    "csw_r" : 0,
    "csw_t" : 0,
    "xi_0" : 1,
    "nu" : 1,
    "isAnisotropic" : False,
    "boundary_phases" : [ 1.0, 1.0, 1.0, 1.0 ]
}))

# cheby
c=g.algorithms.polynomials.chebyshev({
    "low"   : 0.08,
    "high"  : 2.0,
    "order" : 10,
})

# implicitly restarted lanczos
irl=g.algorithms.iterative.irl({
    "Nk" : 60,
    "Nstop" : 60,
    "Nm" : 80,
    "resid" : 1e-8,
    "betastp" : 1e-8,
    "maxiter" : 20,
    "Nminres" : 7,
})

# start vector
start=g.vspincolor(w.grid_eo)
start[:]=g.vspincolor([[1,1,1],[1,1,1],[1,1,1],[1,1,1]])

# generate eigenvectors
ok,evec,ev=irl(c(w.NDagN), start)
assert(ok)

# memory info
g.meminfo()

# print eigenvalues of NDagN as well
for i,v in enumerate(evec):
    w.NDagN(v,start)
    g.message("%d %g %g" % (i,g.innerProduct(v,start).real,ev[i]))
