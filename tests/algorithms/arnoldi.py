#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

# load configuration
U = g.qcd.gauge.random(g.grid([8, 8, 8, 8], g.single), g.random("test"))

# wilson
w = g.qcd.fermion.wilson_clover(
    U,
    {
        "kappa": 0.137,
        "csw_r": 0,
        "csw_t": 0,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)

expected_largest_eigenvalue = 7.438661573807548 - 0.009147480237836585j

# start vector
start = g.vspincolor(w.F_grid)
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])

# arnoldi with modest convergence criterion
a = g.algorithms.eigen.arnoldi(Nmin=50, Nmax=120, Nstep=10, Nstop=1, resid=1e-5)
evec, evals = a(w, start)

# expect the largest eigenvector to have converged somewhat
evals_test = g.algorithms.eigen.evals(w, evec[-1:], check_eps2=1e5 * evals[-1] ** 2.0)
assert abs(evals_test[-1] - expected_largest_eigenvalue) < 1e-3
