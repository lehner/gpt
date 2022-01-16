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
U = g.load("/hpcgpfs01/work/clehner/configs/openQCD/A250t000n54")

# do everything in single-precision
U = g.convert(U, g.single)

# use the gauge configuration grid
grid = U[0].grid

# quark
w = g.qcd.fermion.wilson_clover(
    U,
    {
        "kappa": 0.13565,
        "csw_r": 2.0171,
        "csw_t": 2.0171,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)

# create point source
src = g.mspincolor(grid)
g.create.point(src, [0, 0, 0, 0])

# even-odd preconditioned matrix
eo = g.qcd.fermion.preconditioner.eo2_ne(parity=g.even)

# cheby
c = g.algorithms.polynomial.chebyshev({"low": 0.0005, "high": 3.5, "order": 50})

# implicitly restarted lanczos
irl = g.algorithms.eigen.irl(
    {
        "Nk": 60,
        "Nstop": 60,
        "Nm": 80,
        "resid": 1e-8,
        "betastp": 0.0,
        "maxiter": 100,
        "Nminres": 0,
        #    "maxapply" : 100
    }
)

# start vector
path_to_evec = "/hpcgpfs01/scratch/clehner/openQCD/evec"
try:
    evec, ev = g.load(path_to_evec, {"grids": w.F_grid_eo})
except g.LoadError:
    start = g.vspincolor(w.F_grid_eo)
    start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])

    # generate eigenvectors
    evec, ev_cheby = irl(
        c(eo(w).NDagN),
        start,
        g.checkpointer("/hpcgpfs01/scratch/clehner/openQCD/checkpoint"),
    )
    ev, eps2 = g.algorithms.eigen.evals(eo(w).Mpc, evec, real=True)
    assert eps2 < 1e-8

    # save eigenvectors
    g.save(path_to_evec, [evec, ev])

# build solver
inv = g.algorithms.inverter
dcg = inv.sequence(inv.deflate(evec, ev), inv.cg({"eps": 1e-6, "maxiter": 1000}))
slv = w.propagator(inv.preconditioned(eo, dcg))

# propagator
dst = g.mspincolor(grid)
slv(dst, src)

# two-point
correlator = g.slice(g.trace(dst * g.adj(dst)), 3)

# output
for t, c in enumerate(correlator):
    g.message(t, c.real)
