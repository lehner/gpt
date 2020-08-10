#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import sys

# load configuration
U = g.qcd.gauge.random(g.grid([8, 8, 8, 8], g.single), g.random("test"))

# wilson, eo prec
parity = g.odd
w = g.qcd.fermion.preconditioner.eo1_ne(parity=parity)(
    g.qcd.fermion.wilson_clover(
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
)


# cheby
c = g.algorithms.polynomial.chebyshev({"low": 0.5, "high": 2.0, "order": 10})

# implicitly restarted lanczos
irl = g.algorithms.eigen.irl(
    {
        "Nk": 60,
        "Nstop": 60,
        "Nm": 80,
        "resid": 1e-8,
        "betastp": 0.0,
        "maxiter": 20,
        "Nminres": 7,
        #    "maxapply" : 100
    }
)

# start vector
start = g.vspincolor(w.F_grid_eo)
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
start.checkerboard(parity)

# generate eigenvectors
evec, ev = irl(c(w.Mpc), start)  # , g.checkpointer("checkpoint")

# memory info
g.mem_report()

# print eigenvalues of NDagN as well
evals = g.algorithms.eigen.evals(w.Mpc, evec, check_eps2=1e-11, real=True)

# deflated solver
cg = g.algorithms.inverter.cg({"eps": 1e-6, "maxiter": 1000})
defl = g.algorithms.eigen.deflate(cg, evec, evals)

sol_cg = g.eval(cg(w.Mpc) * start)
eps2 = g.norm2(w.Mpc * sol_cg - start) / g.norm2(start)
niter_cg = len(cg.history)
g.message("Test resid/iter cg: ", eps2, niter_cg)
assert eps2 < 1e-8

sol_defl = g.eval(defl(w.Mpc) * start)
eps2 = g.norm2(w.Mpc * sol_defl - start) / g.norm2(start)
niter_defl = len(cg.history)
g.message("Test resid/iter deflated cg: ", eps2, niter_defl)
assert eps2 < 1e-8

assert niter_defl < niter_cg

# block
grid_coarse = g.block.grid(w.F_grid_eo, [2, 2, 2, 2])
nbasis = 20
cstart = g.vcomplex(grid_coarse, nbasis)
cstart[:] = g.vcomplex([1] * nbasis, nbasis)
basis = evec[0:nbasis]
for i in range(2):
    g.block.orthonormalize(grid_coarse, basis)

# define coarse-grid operator
cop = g.block.operator(c(w.Mpc), grid_coarse, basis)
tmp = g.lattice(cstart)
tmpf = g.lattice(basis[0])
tmp @= cop * cstart
g.block.promote(tmp, tmpf, basis)
g.block.project(cstart, tmpf, basis)
eps2 = g.norm2(cstart - tmp) / g.norm2(cstart)
g.message(f"Test coarse-grid promote/project cycle: {eps2}")
assert eps2 < 1e-13

# coarse-grid lanczos
cevec, cev = irl(cop, cstart)

# smoothened evals
smoother = g.algorithms.inverter.cg({"eps": 1e-6, "maxiter": 10})(w.Mpc)
smoothed_evals = []
g.default.push_verbose("cg", False)
for i, cv in enumerate(cevec):
    g.block.promote(cv, tmpf, basis)
    tmpf @= smoother * tmpf
    smoothed_evals = smoothed_evals + g.algorithms.eigen.evals(
        w.Mpc, [tmpf], check_eps2=1, real=True
    )
g.default.pop_verbose()

# test coarse-grid deflation (re-use fine-grid evals instead of smoothing)
cdefl = g.algorithms.eigen.coarse_deflate(cg, cevec, basis, smoothed_evals)

sol_cdefl = g.eval(cdefl(w.Mpc) * start)
eps2 = g.norm2(w.Mpc * sol_cdefl - start) / g.norm2(start)
niter_cdefl = len(cg.history)
g.message("Test resid/iter coarse-grid deflated cg: ", eps2, niter_cdefl)
g.message("Compare fine-grid deflated cg iter: ", niter_defl)
assert eps2 < 1e-8

assert niter_cdefl < niter_cg
