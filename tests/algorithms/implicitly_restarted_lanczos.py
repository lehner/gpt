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
c = g.algorithms.polynomial.chebyshev({"low": 0.13, "high": 2.0, "order": 10})

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
        "sort_eigenvalues": lambda x: sorted(x),
    }
)

bl = g.algorithms.eigen.block_lanczos(maxiter=200, Nstop=60, miniter=30, step=10, resid=1e-8)

# start vector
start = g.vspincolor(w.Mpc.vector_space[0].grid)
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
start.checkerboard(parity)

# generate eigenvectors
evec, ev = irl(c(w.Mpc), start)  # , g.checkpointer("checkpoint")

bstart_block = [g.vspincolor(w.Mpc.vector_space[0].grid) for _ in range(4)]
g.random("test").cnormal(bstart_block)
for b in bstart_block:
    b.checkerboard(parity)
evc0, ev0 = bl(c(w.Mpc), bstart_block)
assert len(ev0) == len(evc0)
assert len(ev0) == 60

ev1, eps2 = g.algorithms.eigen.evals(c(w.Mpc), evc0, real=True)

for i in range(60):
    eps = abs(ev0[i]/ev[i] - 1)
    assert eps < 1e-5
    eps = abs(ev1[i]/ev0[i] - 1)
    assert eps < 1e-5

# memory info
g.mem_report()

# print eigenvalues of NDagN as well
evals, eps2 = g.algorithms.eigen.evals(w.Mpc, evec, real=True)
assert all([e2 < 1e-11 for e2 in eps2])

# test low-mode approximation of inverse
inv = g.algorithms.inverter
lma = inv.deflate(evec, evals)(w.Mpc)
for i in range(len(evals)):
    eps2 = g.norm2(evals[i] * lma * evec[i] - evec[i]) / g.norm2(evec[i]) * evals[i]
    g.message(f"Test low-mode approximation for evec[{i}]: {eps2}")
    assert eps2 < 5e-10

# deflated solver
cg = inv.cg({"eps": 1e-6, "maxiter": 1000})
defl = inv.sequence(inv.deflate(evec, evals), cg)
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
grid_coarse = g.block.grid(w.Mpc.vector_space[0].grid, [2, 2, 2, 2])
nbasis = 20
cstart = g.vcomplex(grid_coarse, nbasis)
cstart[:] = g.vcomplex([1] * nbasis, nbasis)
basis = evec[0:nbasis]
b = g.block.map(grid_coarse, basis)
for i in range(2):
    b.orthonormalize()

# define coarse-grid operator
cop = b.coarse_operator(c(w.Mpc))
eps2 = g.norm2(cop * cstart - b.project * c(w.Mpc) * b.promote * cstart) / g.norm2(cstart)
g.message(f"Test coarse-grid promote/project cycle: {eps2}")
assert eps2 < 1e-13

# coarse-grid lanczos
cevec, cev = irl(cop, cstart)

# smoothened evals
smoother = inv.cg({"eps": 1e-6, "maxiter": 10})(w.Mpc)
smoothed_evals = []
g.default.push_verbose("cg", False)
tmpf = g.lattice(basis[0])
for i, cv in enumerate(cevec):
    tmpf @= smoother * b.promote * cv
    smoothed_evals = smoothed_evals + g.algorithms.eigen.evals(
        w.Mpc, [tmpf], calculate_eps2=False, real=True
    )
g.default.pop_verbose()

# test coarse-grid deflation (re-use fine-grid evals instead of smoothing)
cdefl = inv.sequence(inv.coarse_deflate(cevec, basis, smoothed_evals), cg)

sol_cdefl = g.eval(cdefl(w.Mpc) * start)
eps2 = g.norm2(w.Mpc * sol_cdefl - start) / g.norm2(start)
niter_cdefl = len(cg.history)
g.message("Test resid/iter coarse-grid deflated cg: ", eps2, niter_cdefl)
g.message("Compare fine-grid deflated cg iter: ", niter_defl)
g.message("Compare cg iter: ", niter_cg)
assert eps2 < 1e-8

assert niter_cdefl < niter_cg
