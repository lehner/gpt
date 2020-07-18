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

# start vector
start = g.vspincolor(w.F_grid)
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])

# power iteration for comparison of largest eigenvalue
#pi = g.algorithms.eigen.power_iteration({"eps": 1e-5, "maxiter": 300})
#pi(w, start)

expected_largest_eigenvalue = 7.438443531886123 + 0.013717311016902744j

# arnoldi iteration
a = g.algorithms.eigen.arnoldi_iteration(w, start)
for i in range(12):
    for j in range(10):
        a() # single arnoldi step

    evals, little_evec = a.little_eig()
    g.message("eval_arnoldi_max[",i,"] =",evals[-1].real)

assert abs(evals[-1] - expected_largest_eigenvalue) < 1e-3

# obtain eigenvectors
evec = a.rotate_basis_to_evec(little_evec)

# expect the largest eigenvector to have converged somewhat
evals_test = g.algorithms.eigen.evals(w, evec[-1:], check_eps2=1e-3)

# TODO: implement eigen.arnoldi class with same interface as eigen.irl

