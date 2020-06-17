#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Exercise linear solvers
#
import gpt as g
import numpy as np
import sys
import time
import os.path

# load configuration
precision=g.double
U=g.qcd.gauge.random(g.grid([8,8,8,16],precision),g.random("test"))

# use the gauge configuration grid
grid=U[0].grid

# quark
w=g.qcd.fermion.wilson_clover(U,{
    "kappa" : 0.13565,
    "csw_r" : 2.0171 / 2., # for now test with very heavy quark
    "csw_t" : 2.0171 / 2.,
    "xi_0" : 1,
    "nu" : 1,
    "isAnisotropic" : False,
    "boundary_phases" : [ 1.0, 1.0, 1.0, 1.0 ]
})

# create point source
src=g.vspincolor(grid)
src[0,1,0,0]=g.vspincolor([ [1]*3 ] * 4)

# build solvers
s = g.qcd.fermion.solver
a = g.algorithms.iterative
w_sp=w.converted(g.single)
eo2_odd=g.qcd.fermion.preconditioner.eo2(w, parity = g.odd)
eo2_even=g.qcd.fermion.preconditioner.eo2(w, parity = g.even)
eo1_odd=g.qcd.fermion.preconditioner.eo1(w, parity = g.odd)
eo1_even=g.qcd.fermion.preconditioner.eo1(w, parity = g.even)
eo2_odd_sp=g.qcd.fermion.preconditioner.eo2(w_sp, parity = g.odd)
# default
eo2=eo2_odd
eo2_sp=eo2_odd_sp

# run with higher stopping condition since it will be the reference run
slv_cg = s.propagator(
    s.inv_eo_ne(eo2,
            a.cg({
                "eps": 1e-8,
                "maxiter": 1000
            })))
# other pc and parity
slv_cg_eo2_even = s.propagator(
    s.inv_eo_ne(eo2_even,
            a.cg({
                "eps": 1e-8,
                "maxiter": 1000
            })))
slv_cg_eo1_odd = s.propagator(
    s.inv_eo_ne(eo1_odd,
            a.cg({
                "eps": 1e-8,
                "maxiter": 1000
            })))
slv_cg_eo1_even = s.propagator(
    s.inv_eo_ne(eo1_even,
            a.cg({
                "eps": 1e-8,
                "maxiter": 1000
            })))
# other parity/pc
slv_cg = s.propagator(
    s.inv_eo_ne(eo2,
            a.cg({
                "eps": 1e-8,
                "maxiter": 1000
            })))

# solvers to test against CG
slv_mr = s.propagator(
    s.inv_eo_ne(eo2,
        a.mr({
            "eps": 1e-6,
            "maxiter": 1000,
            "relax": 1.0
        })))
slv_bicgstab = s.propagator(
    s.inv_eo_ne(eo2,
            a.bicgstab({
                "eps": 1e-6,
                "maxiter": 1000
            })))
slv_fgcr = s.propagator(
    s.inv_eo_ne(eo2,
        a.fgcr({
            "eps": 1e-6,
            "maxiter": 1000,
            "restartlen": 20
        })))
slv_fgmres = s.propagator(
    s.inv_eo_ne(eo2,
        a.fgmres({
            "eps": 1e-6,
            "maxiter": 1000,
            "restartlen": 20
        })))

# defect-correcting solver at the full field level
slv_dci=s.propagator(
    a.defect_correcting_inverter(
        s.inv_eo_ne(
            eo2,
            a.cg({
                "eps": 1e-40,
                "maxiter": 25
            })),
        eps = 1e-6,maxiter=10)(w.M),w)

# defect-correcting solver at the even-odd level
slv_dci_eo=s.propagator(
        s.inv_eo_ne(
            eo2,
            a.defect_correcting_inverter(
                a.cg({
                    "eps": 1e-40,
                    "maxiter": 25
                })(eo2.NDagN),
                eps = 1e-6,maxiter=10)),w)

# mixed-precision defect-correcting solver at the full field level
slv_dci_mp=s.propagator(
    a.defect_correcting_inverter(
        s.inv_eo_ne(
            eo2_sp,
            a.cg({
                "eps": 1e-40,
                "maxiter": 25
            })).converted(g.double, verbose = True),
        eps = 1e-6,maxiter=10)(w.M),w)

# perform solves (reference)
dst_cg = g.eval( slv_cg * src )
g.message("CG finished")

timings={}
resid={}
def test(slv,name):
    t0=g.time()
    dst = g.eval( slv * src )
    t1=g.time()
    eps2=g.norm2(dst_cg-dst) / g.norm2(dst_cg)
    g.message("%s finished: eps^2(CG) = %g" % (name,eps2))
    timings[name] = t1-t0
    resid[name] = eps2 ** 0.5
    assert(eps2 < 1e-7)

test(slv_cg_eo2_even,"CG eo2_even")
test(slv_cg_eo1_even,"CG eo1_even")
test(slv_cg_eo1_odd,"CG eo1_odd")
test(slv_dci,"Defect-correcting solver")
test(slv_dci_eo,"Defect-correcting (eo)")
test(slv_dci_mp,"Defect-correcting (mixed-precision)")
test(slv_mr,"MR")
test(slv_bicgstab,"BICGSTAB")
test(slv_fgcr,"FGCR")
test(slv_fgmres,"FGMRES")

# summary
g.message("--------------------------------------------------------------------------------")
g.message("                            Summary of solver tests")
g.message("--------------------------------------------------------------------------------")
g.message("%-38s %-25s %-25s" % ("Solver name","Solve time / s","Difference with CG result"))
for t in timings:
    g.message("%-38s %-25s %-25s" % (t,timings[t],resid[t]))
