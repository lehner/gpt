#!/usr/bin/env python3
#
# Authors: Christoph Lehner
# Title: Minimal RMHMC example
#
# H(p,q) = 1/2 p^2 + S(q)
# p = M^{-1/2} P ; H = 1/2 P^T (M^{-1/2})^T M^{-1/2} P + S(q) - log(det(M^{-1/2}))
# -> G^{-1} = M^{-1/2}^T M^{-1/2}
#    H = 1/2 P^T G^{-1} P + S(q) - (1/2) log(det(M^{-1/2})) - (1/2) log(det(M^{-1/2}^T))
#      = 1/2 P^T G^{-1} P + S(q) - (1/2) log(det(G^{-1}))
#      = 1/2 P^T G^{-1} P + S(q) + (1/2) log(det(G))
#
# see also (1.1) of https://arxiv.org/pdf/2412.19904
#
# We can then introduce an auxiliary real degree of freedom R to write
#
# H = 1/2 P^T G^{-1} P + S(q) + 1/2 R^T G R
#
import gpt as g
import numpy as np
import sys, os

# parameters
tau = g.default.get_float("--tau", 1.0)
root = g.default.get("--root", None)
beta = g.default.get_float("--beta", 2.95)
seed = g.default.get("--seed", "hmc-pure-gauge")
n = g.default.get_int("--n", 1000)

# grid
grid = g.grid([4, 4, 4, 8], g.double)
rng = g.random(seed)

# load state / initialize state
U = g.qcd.gauge.unit(grid)
rng.normal_element(U, scale=0.3)

fn_try = None
i0 = 0
for i in range(0, n):
    fn = f"{root}/ckpoint_lat.{i}"
    if os.path.exists(fn):
        fn_try = fn
        i0 = i + 1

if fn_try is not None:
    rng = g.random(fn_try)
    U0 = g.load(fn_try)
    for mu in range(4):
        U[mu] @= U0[mu]

p_mom = g.group.cartesian(U)  # conjugate momenta
r_mom = g.group.cartesian(U)  # auxiliary fields

##################################################################
# Define mass operator
# G^{-1} = f_lap_U_sqr, should be fast
# make it a polynomial of the laplace
##################################################################

# f(lap) = scale * (lap + mass_1) * (lap + mass_2) * ... * (lap + mass_n)

f_lap = g.qcd.gauge.algebra_laplace_polynomial(U, 1.0, [-0.1, -0.2, -0.3])  # scale  # masses

f_lap = g.qcd.gauge.algebra_laplace_polynomial(U, 1.0, [-0.1])  # scale  # masses

g.default.push_verbose("block_cg", True)
cg = g.algorithms.inverter.block_cg({"eps": 1e-15, "maxiter": 300})

inv_f_lap = f_lap.inverse(cg)

f_lap_U = g.matrix_operator(
    mat=f_lap, inv_mat=inv_f_lap, accept_list=True, accept_guess=(False, True)
)

inv_f_lap_U = f_lap_U.inv()

f_lap_U_sqr = f_lap_U * f_lap_U
inv_f_lap_U_sqr = inv_f_lap_U * inv_f_lap_U


def inv_f_lap_sqr_projected_gradient(U, vec):
    # dM^{-2} = d(M^{-1}) M^{-1} + M^{-1} d(M^{-1}) = - M^{-1} dM M^{-2} - M^{-2} dM M^{-1}
    invM_vec = g(inv_f_lap_U * (U + vec))[-len(vec) :]
    invM2_vec = g(inv_f_lap_U * (U + invM_vec))[-len(vec) :]

    grad1 = f_lap.projected_gradient(invM_vec, U, invM2_vec)
    return [g(-2 * x) for x in grad1]


def f_lap_sqr_projected_gradient(U, vec):
    # d(M^2) = M dM + dM M
    M_vec = g(f_lap_U * (U + vec))[-len(vec) :]
    grad1 = f_lap.projected_gradient(M_vec, U, vec)
    return [g(2 * x) for x in grad1]


# P-action needs to be on inner SW timescale and should be polynomial
aP = g.qcd.scalar.action.general_mass_term(
    inv_M=f_lap_U_sqr, sqrt_inv_M=f_lap_U, inv_M_projected_gradient=f_lap_sqr_projected_gradient
)

# R-action can be done less frequently and can be inverse of polynomial
aR = g.qcd.scalar.action.general_mass_term(
    inv_M=f_lap_U_sqr.inv(),
    sqrt_inv_M=f_lap_U.inv(),
    inv_M_projected_gradient=inv_f_lap_sqr_projected_gradient,
)

##################################################################

# test action
rng.element(p_mom)
aP.assert_gradient_error(rng, U + p_mom, U + p_mom, 1e-3, 1e-8)

rng.element(r_mom)
aR.assert_gradient_error(rng, U + r_mom, U + r_mom, 1e-3, 1e-8)


# wilson action
aQ = g.qcd.gauge.action.iwasaki(beta)

# test integration
aP.draw(U + p_mom, rng)
aR.draw(U + r_mom, rng)

sympl = g.algorithms.integrator.symplectic


def hamiltonian():
    a_gauge = aQ(U)
    a_p_mom = aP(U + p_mom)
    a_r_mom = aR(U + r_mom)
    return a_gauge + a_p_mom + a_r_mom


# def add(a, b):
#    return [g(x+y) for x, y in zip(a, b)]


def add(a, b, c):
    return [g(x + y + z) for x, y, z in zip(a, b, c)]


# create OMF2 and OMF2_force_gradient integrators
p_mom2 = g.copy(p_mom)
U2 = g.copy(U)
_iq_imp = sympl.update_q(U, lambda: aP.gradient(U2 + p_mom, p_mom), "Q")
_ip_imp = sympl.update_p(
    p_mom,
    lambda: add(aQ.gradient(U, U), aP.gradient(U + p_mom2, U), aR.gradient(U + r_mom, U)),
    "P",
)
ip_imp = sympl.implicit_update(p_mom, p_mom2, _ip_imp, eps=1e-15)
iq_imp = sympl.implicit_update(U, U2, _iq_imp, eps=1e-15)
ip_fg_imp = sympl.implicit_update(
    U + p_mom,
    U2 + p_mom2,
    sympl.update_p_force_gradient(U + U2, _iq_imp, p_mom + p_mom2, _ip_imp, _ip_imp, "P_FG"),
    eps=1e-15,
)

integrators = [
    sympl.leap_frog(1, ip_imp, iq_imp),
    sympl.OMF2(1, ip_imp, iq_imp),
    sympl.OMF2(1, ip_imp, iq_imp, 1.0 / 6.0),
    sympl.OMF2(1, ip_imp, iq_imp, 1.0 / 6.0, 1.0 / 6.0),
    sympl.OMF4(1, ip_imp, iq_imp),
    sympl.OMF2_force_gradient(1, ip_imp, iq_imp, ip_fg_imp, 0.18, 0.5),
    sympl.OMF2_force_gradient(1, ip_imp, iq_imp, ip_fg_imp),
]

for integrator in integrators:
    g.message(integrator)

# test error scaling of integrator
h0 = hamiltonian()

U0 = g.copy(U)
p_mom0 = g.copy(p_mom)
for dt in [1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0]:
    for integrator in integrators:
        g.copy(U, U0)
        g.copy(p_mom, p_mom0)
        integrator(dt)
        h1 = hamiltonian()
        integrator(-dt)
        eps2 = sum([g.norm2(u - u2) for u, u2 in zip(U + p_mom, U0 + p_mom0)])
        g.message(f"dH dt={dt} {integrator.__name__} dH={h1-h0:.3g} reversibility={eps2**0.5:.2g}")
