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

# arnoldi with modest convergence criterion
a = g.algorithms.eigen.arnoldi(Nmin=50, Nmax=120, Nstep=10, Nstop=4, resid=1e-5)
ira = g.algorithms.eigen.arnoldi(
    Nmin=50, Nmax=120, Nstep=10, Nstop=4, resid=1e-5, implicit_restart=True
)


def test(a, name):
    t0 = g.time()
    evec, evals = a(w, start)
    t1 = g.time()
    evals_test, evals_eps2 = g.algorithms.eigen.evals(w, evec)

    largest_eval = 7.437

    g.message(f"{name} finished in {t1-t0} s")

    for i in range(len(evals_eps2)):
        assert evals_eps2[i] / largest_eval**2.0 < 1e-5
        assert abs(evals_test[i] - evals[i]) < 1e-6


# expect the largest eigenvector to have converged somewhat
test(a, "Arnoldi")
test(ira, "Implicitly Restarted Arnoldi")
