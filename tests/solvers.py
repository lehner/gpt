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
#homedir = os.path.expanduser("~")
#U = g.load(homedir + "/configs/openqcd/test_16x8_pbcn6")
U = g.load("/hpcgpfs01/work/clehner/configs/32IDfine/ckpoint_lat.200") # TODO: add parallel RNG so we can do tests from random gauge configs

# do everything in single-precision
U = g.convert(U, g.single)

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
src=g.mspincolor(grid)
g.create.point(src, [0,0,0,0])

# build solvers
s = g.qcd.fermion.solver
slv_cg = s.propagator(
    s.eo_ne(g.qcd.fermion.preconditioner.eo2(w),
            g.algorithms.iterative.cg({
                "eps": 1e-6,
                "maxiter": 1000
            })))
slv_mr = s.propagator(
    s.eo_ne(
        g.qcd.fermion.preconditioner.eo2(w),
        g.algorithms.iterative.mr({
            "eps": 1e-6,
            "maxiter": 1000,
            "relax": 1.0
        })))
slv_bicgstab = s.propagator(
    s.eo_ne(g.qcd.fermion.preconditioner.eo2(w),
            g.algorithms.iterative.bicgstab({
                "eps": 1e-6,
                "maxiter": 1000
            })))
slv_fgcr = s.propagator(
    s.eo_ne(
        g.qcd.fermion.preconditioner.eo2(w),
        g.algorithms.iterative.fgcr({
            "eps": 1e-6,
            "maxiter": 1000,
            "restartlen": 20
        })))
slv_fgmres = s.propagator(
    s.eo_ne(
        g.qcd.fermion.preconditioner.eo2(w),
        g.algorithms.iterative.fgmres({
            "eps": 1e-6,
            "maxiter": 1000,
            "restartlen": 20
        })))

# rhs vectors
dst_cg=g.mspincolor(grid)
dst_mr=g.mspincolor(grid)
dst_bicgstab=g.mspincolor(grid)
dst_fgcr=g.mspincolor(grid)
dst_fgmres=g.mspincolor(grid)

# perform solves
slv_cg(src, dst_cg)
g.message("CG finished")
slv_mr(src, dst_mr)
g.message("MR finished: eps^2(CG) = %g" % g.norm2(dst_cg-dst_mr))
slv_bicgstab(src, dst_bicgstab)
g.message("BICGSTAB finished: eps^2(CG) = %g" % g.norm2(dst_cg-dst_bicgstab))
slv_fgcr(src, dst_fgcr)
g.message("FGCR finished: eps^2(CG) = %g" % g.norm2(dst_cg-dst_fgcr))
slv_fgmres(src, dst_fgmres)
g.message("FGMRES finished: eps^2(CG) = %g" % g.norm2(dst_cg-dst_fgmres))
