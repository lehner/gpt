#!/usr/bin/env python3
#
# Authors: Christoph Lehner
#
import gpt as g


# gauge field
rng = g.random("test")
U = g.qcd.gauge.random(g.grid([8, 8, 8, 8], g.single), rng)
g.message("Plaquette:", g.qcd.gauge.plaquette(U))

mobius_params = {
    "mass": 0.08,
    "M5": 1.8,
    "b": 1.5,
    "c": 0.5,
    "Ls": 12,
    "boundary_phases": [1.0, 1.0, 1.0, 1.0],
}

w = g.qcd.fermion.mobius(U, mobius_params)


# shortcuts
inv = g.algorithms.inverter
inv_pc = inv.preconditioned
pc = g.qcd.fermion.preconditioner


# point source
ssrc = g.vspincolor(w.F_grid_eo)
rng.cnormal(ssrc)

dst2 = g.lattice(ssrc)

# TODO: first test correctness

# use different solver and compare
g.default.push_verbose("cg_convergence", True)

ssrc.checkerboard(g.odd)
matrix = pc.eo2_ne(parity=g.odd)(w).Mpc

cg1 = inv.cg({"eps": 1e-6, "maxiter": 250})
inv2_w = cg1(matrix)

icg = inv.cg({"eps": 1e-7, "maxiter": 4})
icg.verbose_convergence = False
icg.verbose = False
open_inv = g.qcd.fermion.preconditioner.open_boundary_local(icg, margin=2)

# check that matrix is still Hermitian
x = g.inner_product(ssrc, open_inv(matrix) * ssrc)
assert abs(x.imag / x.real) < 1e-7

cg2 = inv.cg(eps=1e-6, maxiter=130, prec=open_inv)
inv3_w = cg2(matrix)

dst2 = g.eval(inv2_w * ssrc)
dst3 = g.eval(inv3_w * ssrc)

eps2 = g.norm2(dst2 - dst3) / g.norm2(dst2)
g.message(f"Both solutions agree to: eps^2 = {eps2}")
assert eps2 < 1e-10

speedup = len(cg1.history) / len(cg2.history)
g.message(f"Speedup in terms of outer CG iterations: {speedup}")
assert speedup > 1.0
