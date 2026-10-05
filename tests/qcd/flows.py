#!/usr/bin/env python3
#
# Flow-type models
#

import gpt as g
import numpy as np

# general setup
rng = g.random("test")

#U = g.qcd.gauge.random(g.grid([8*2,8*2,8*4,8*4], g.double), rng)
U = g.qcd.gauge.random(g.grid([8,8,8,8], g.double), rng)
rad = g.ad.reverse

# specific even/odd pattern
even, odd = g.even_odd_projectors(U[0].grid)
full = g(even + odd)
none = g(0 * full)

# plaquette-type flow (P-FTHMC with local weights)
rho = g.complex(even.grid)

# regress = True
regress = False

L = np.array(even.grid.gdimensions)
coor = g.coordinates(even)
coor = (coor + L//2) % L - L//2
r2 = np.sum(coor*coor, axis=1)

if regress:
    rho[:] = 0.12
else:
    rho[:] = 0.12*np.exp(-r2)

params = [rho]

description = [
    (
    [(rho, g.path().f(nu).f(mu).b(nu).b(mu)) for nu in range(4) if mu != nu]
    +
    [(rho, g.path().b(nu).f(mu).f(nu).b(mu)) for nu in range(4) if mu != nu]
    )
    for mu in range(4)
]

pt_e = [
    g.qcd.gauge.smear.directional_parallel_transport(
        U,
        description[j],
        j,
        full,
        even,
        params
    )
    for j in range(4)
]

pt_o = [
    g.qcd.gauge.smear.directional_parallel_transport(
        U,
        description[j],
        j,
        full,
        odd,
        params
    )
    for j in range(4)
]


# first establish agreement with specialized local_stout
if regress:
    Uprime0 = pt_e[1](U + params)[0:4]
    Uprime1 = g.qcd.gauge.smear.local_stout(rho=0.12, dimension=1, checkerboard=g.even)(U)
    eps2 = sum(g.norm2(x - y) / g.norm2(x) for x, y in zip(Uprime0, Uprime1))
    g.message(f"Test agreement with P-FTHMC: {eps2}")
    assert eps2 < 1e-28

# next test that it works in the Jacobian
# version with separate parameters for the layers
a1 = g.qcd.gauge.action.iwasaki(6)
a1 = a1.transformed(pt_o[0], indices=[0,1,2,3,4], projection=[0,1,2,3])
a1 = a1.transformed(pt_e[0], indices=[0,1,2,3,5], projection=[0,1,2,3,4])
a1 = a1.transformed(pt_o[1], indices=[0,1,2,3,6], projection=[0,1,2,3,4,5])
rho1=g.copy(rho)
rho2=g.copy(rho)
rho3=g.copy(rho)
params2=[rho1,rho2,rho3]

a1.assert_gradient_error(rng, U + params2, U + params2, 1e-4, 1e-7)


# version with separate parameters for the layers
a1 = g.qcd.gauge.action.iwasaki(6)
a1 = a1.transformed(pt_o[0], indices=[0,1,2,3,4], projection=[0,1,2,3])
a1 = a1.transformed(pt_e[0], indices=[0,1,2,3,4], projection=[0,1,2,3,4])
a1 = a1.transformed(pt_o[1], indices=[0,1,2,3,4], projection=[0,1,2,3,4])
rho1=g.copy(rho)
params2=[rho1]

a1.assert_gradient_error(rng, U + params2, U + params2, 1e-4, 1e-7)


# now test the local log-det-jacobian
act1 = g.qcd.gauge.smear.local_stout(rho=0.12, dimension=0, checkerboard=g.odd).action_log_det_jacobian()
act2 = pt_o[0].action_log_det_jacobian()

if regress:
    v1 = act1(U)
    v2 = act2(U + params)
    eps = abs(v1 - v2) / abs(v1)
    g.message(f"Log-det-jacobian agreement: {eps}")
    assert eps < 1e-13


# finally check the force terms of the log-det-jacobian
act2.assert_gradient_error(rng, U + params, U + params, 1e-4, 1e-7)


# a non-trivial function of the weighted loop sum, U_mu' = exp(TA(P1 f(sm))) U_mu
def loop_function(sm, xparams):
    return sm + sm * sm * 0.5 + sm * g.adj(sm) * sm * 0.25


def make_pt(mu, P1, f, generic=False, params=params):
    pt = g.qcd.gauge.smear.directional_parallel_transport(
        U, description[mu], mu, full, P1, params, loop_function=f
    )
    if generic:
        # force the generic graph path (no staple factorization)
        pt.description_staple = None
    return pt


# the identity as loop function reproduces the default
pt_f = make_pt(0, odd, loop_function)
Uprime0 = make_pt(0, odd, None)(U + params)[0:4]
Uprime1 = make_pt(0, odd, lambda sm, xparams: sm)(U + params)[0:4]
eps2 = sum(g.norm2(x - y) / g.norm2(x) for x, y in zip(Uprime0, Uprime1))
g.message(f"Identity loop function agreement: {eps2}")
assert eps2 < 1e-28
Uprime1 = pt_f(U + params)[0:4]
eps2 = g.norm2(Uprime0[0] - Uprime1[0]) / g.norm2(Uprime0[0])
g.message(f"Loop function changes the flow: {eps2}")
assert eps2 > 1e-8

# local and generic paths agree
pt_f_generic = make_pt(0, odd, loop_function, generic=True)
M0 = pt_f.jacobian_matrix(U + params)
M1 = pt_f_generic.jacobian_matrix(U + params)
eps2 = g.norm2(M0 - M1) / g.norm2(M0)
g.message(f"Loop function: local vs generic Jacobian block: {eps2}")
assert eps2 < 1e-25

v0 = pt_f.log_det_jacobian(U + params)
v1 = pt_f_generic.log_det_jacobian(U + params)
eps = abs(v0 - v1) / abs(v0)
g.message(f"Loop function: local vs generic log-det: {eps}")
assert eps < 1e-12

gr0 = pt_f.action_log_det_jacobian().gradient(U + params, U + params)
gr1 = pt_f_generic.action_log_det_jacobian().gradient(U + params, U + params)
for x, y in zip(gr0, gr1):
    eps2 = g.norm2(x - y) / g.norm2(x)
    g.message(f"Loop function: local vs generic log-det force: {eps2}")
    assert eps2 < 1e-20

# inverse
Uprime = pt_f(U + params)
Uinv = pt_f.inv(Uprime[0:4] + params)
eps2 = g.norm2(Uinv[0] - U[0]) / g.norm2(U[0])
g.message(f"Loop function: inverse: {eps2}")
assert eps2 < 1e-25

# Jacobian (local VJP and generic graph) in a flow
for generic in [False, True]:
    a1 = g.qcd.gauge.action.iwasaki(6)
    a1 = a1.transformed(
        make_pt(0, odd, loop_function, generic), indices=[0, 1, 2, 3, 4], projection=[0, 1, 2, 3]
    )
    a1 = a1.transformed(
        make_pt(0, even, loop_function, generic), indices=[0, 1, 2, 3, 4], projection=[0, 1, 2, 3, 4]
    )
    a1.assert_gradient_error(rng, U + params, U + params, 1e-4, 1e-7)

# log-det-jacobian force (local and generic)
pt_f.action_log_det_jacobian().assert_gradient_error(rng, U + params, U + params, 1e-4, 1e-7)
pt_f_generic.action_log_det_jacobian().assert_gradient_error(rng, U + params, U + params, 1e-4, 1e-7)

# local_stout passes it through; with a uniform rho = 0.12 the loop sum is
# O(1) everywhere and loop_function above makes the Jacobian singular on some
# sites (no longer a diffeomorphism), so use a weaker function here
act1 = g.qcd.gauge.smear.local_stout(
    rho=0.12, dimension=0, checkerboard=g.odd, loop_function=lambda sm, xparams: sm + sm * sm * 0.05
)
act1.action_log_det_jacobian().assert_gradient_error(rng, U, U, 1e-4, 1e-7)


# a trainable parameter field in the loop function, f(P) = P + p P^2
p = g.complex(even.grid)
rng.normal(p)
p[:] = 0.5 + 0.1 * p[:]
params_p = [rho, p]


def loop_function_p(sm, xparams):
    return sm + sm * sm * xparams[1]


pt_p = make_pt(0, odd, loop_function_p, params=params_p)
pt_p_generic = make_pt(0, odd, loop_function_p, generic=True, params=params_p)

# p = 0 reproduces the identity
p0 = g(0 * p)
Uprime0 = make_pt(0, odd, None)(U + params)[0:4]
Uprime1 = pt_p(U + [rho, p0])[0:4]
eps2 = sum(g.norm2(x - y) / g.norm2(x) for x, y in zip(Uprime0, Uprime1))
g.message(f"Parameter loop function at p = 0: {eps2}")
assert eps2 < 1e-28

# local and generic paths agree, including the parameter gradients
dirs = [rng.normal_element(g.group.cartesian(x)) for x in U + params_p]
for name, gr0, gr1 in [
    (
        "log-det force",
        pt_p.action_log_det_jacobian().gradient(U + params_p, U + params_p),
        pt_p_generic.action_log_det_jacobian().gradient(U + params_p, U + params_p),
    ),
    (
        "Jacobian",
        pt_p.jacobian(U + params_p, pt_p(U + params_p), dirs),
        pt_p_generic.jacobian(U + params_p, pt_p(U + params_p), dirs),
    ),
]:
    for x, y in zip(gr0, gr1):
        eps2 = g.norm2(x - y) / g.norm2(x)
        g.message(f"Parameter loop function: local vs generic {name}: {eps2}")
        assert eps2 < 1e-20

# inverse
Uprime = pt_p(U + params_p)
Uinv = pt_p.inv(Uprime[0:4] + params_p)
eps2 = g.norm2(Uinv[0] - U[0]) / g.norm2(U[0])
g.message(f"Parameter loop function: inverse: {eps2}")
assert eps2 < 1e-25

# gradients w.r.t. links, rho and p (local and generic): a flow of two
# layers sharing the parameters, and the log-det-jacobian
for generic in [False, True]:
    a1 = g.qcd.gauge.action.iwasaki(6)
    a1 = a1.transformed(
        make_pt(0, odd, loop_function_p, generic, params_p),
        indices=[0, 1, 2, 3, 4, 5],
        projection=[0, 1, 2, 3],
    )
    a1 = a1.transformed(
        make_pt(0, even, loop_function_p, generic, params_p),
        indices=[0, 1, 2, 3, 4, 5],
        projection=[0, 1, 2, 3, 4, 5],
    )
    a1.assert_gradient_error(rng, U + params_p, U + params_p, 1e-4, 1e-7)
    # the p-derivative is small compared to the action (and can be O(0.1)
    # along a random direction): a larger step keeps the roundoff of the
    # (4th-order) difference small, the tolerance allows for the cancellation
    a1.assert_gradient_error(rng, U + params_p, [p], 1e-2, 1e-6)

    a2 = (pt_p_generic if generic else pt_p).action_log_det_jacobian()
    a2.assert_gradient_error(rng, U + params_p, U + params_p, 1e-4, 1e-7)
    a2.assert_gradient_error(rng, U + params_p, [p], 3e-3, 1e-7)


# a global number p in f(P) = P + p P^2 (parameters may be numbers): the
# actions and their gradients agree with a site-constant field p, whose
# gradient summed over the sites is the gradient of the number
p_number = 0.5 + 0.1j
p_field = g.complex(even.grid)
p_field[:] = p_number
params_number = [rho, p_number]
params_field = [rho, p_field]


def actions(pt_odd, pt_even):
    a_gauge = g.qcd.gauge.action.iwasaki(6)
    a_gauge = a_gauge.transformed(pt_even, indices=[0, 1, 2, 3, 4, 5], projection=[0, 1, 2, 3])
    a_gauge = a_gauge.transformed(pt_odd, indices=[0, 1, 2, 3, 4, 5], projection=[0, 1, 2, 3, 4, 5])
    return a_gauge, pt_odd.action_log_det_jacobian()


for generic in [False, True]:
    a_number = actions(
        make_pt(0, odd, loop_function_p, generic, params_number),
        make_pt(0, even, loop_function_p, generic, params_number),
    )
    a_field = actions(
        make_pt(0, odd, loop_function_p, generic, params_field),
        make_pt(0, even, loop_function_p, generic, params_field),
    )
    for name, an, af in zip(["gauge", "log det"], a_number, a_field):
        vn, vf = an(U + params_number), af(U + params_field)
        gn = an.gradient(U + params_number, [p_number])[0]
        gf = g.sum(af.gradient(U + params_field, [p_field])[0])
        eps = abs(vn - vf) / abs(vf) + abs(gn - gf) / abs(gf)
        g.message(f"Number vs field parameter ({name}, generic = {generic}): {eps}")
        assert eps < 1e-12
        an.assert_gradient_error(rng, U + params_number, U + params_number, 1e-4, 1e-7)
        an.assert_gradient_error(rng, U + params_number, [p_number], 1e-3, 1e-6)


# the p-gradient of a contraction of the force of the combined action
#   S = S_gauge(Phi_2(Phi_1(U))) - log det J_1(U) - log det J_2(Phi_1(U))
# with Q(U, p) = sum_mu <v_mu, F_mu(U, p)>, F = dS/dU.  Since Q is the
# derivative of S along the group flow U(eps) = exp(eps v) U at eps = 0 and
# mixed partials commute,
#   dQ/dp = d/deps [dS/dp](U(eps), p) at eps = 0,
# which needs only the first-order parameter force.
from gpt.core.group.differentiable_functional import approximation_scheme_4

# (with rho above, the even layer is nearly singular at the origin, where rho
# peaks: det of its Jacobian block is 1e-3 without and ~0 with the loop
# function, so the log det is ill-conditioned there; use a smaller rho)
rho_s = g(0.25 * rho)
params_s = [rho_s, p]
description_s = [(rho_s, path) for _, path in description[0]]


def make_layer(P1):
    return g.qcd.gauge.smear.directional_parallel_transport(
        U, description_s, 0, full, P1, params_s, loop_function=loop_function_p
    )


phi_1 = make_layer(odd)
phi_2 = make_layer(even)
indices = [0, 1, 2, 3, 4, 5]
a_gauge = g.qcd.gauge.action.iwasaki(6)
a_gauge = a_gauge.transformed(phi_2, indices=indices, projection=[0, 1, 2, 3])
a_gauge = a_gauge.transformed(phi_1, indices=indices, projection=indices)
a_ld_1 = phi_1.action_log_det_jacobian()
a_ld_2 = phi_2.action_log_det_jacobian().transformed(phi_1, indices=indices, projection=indices)
a_total = a_gauge + a_ld_1 + a_ld_2
a_total.assert_gradient_error(rng, U + params_s, U + params_s, 1e-4, 1e-7)

v = rng.normal_element(g.group.cartesian(U))
Q = g.group.directional_derivative(a_total, v, along=[0, 1, 2, 3])
dQ_dp = Q.gradient(U + params_s, [p])[0]

# cross-check along a random direction dp with a difference of Q in p
dp = rng.normal_element(g.group.cartesian(p))
a = g.group.inner_product(dp, dQ_dp)
t = 1e-2
b = sum(
    (cc / t) * Q(U + [rho_s, g(g.group.compose(g(dd * t * dp), p))])
    for cc, dd in approximation_scheme_4
)
eps = abs(a - b) / abs(b)
g.message(f"p-gradient of the force contraction: {a} vs {b}, rel {eps}")
assert eps < 1e-7

# and the gradient w.r.t. all fields (links: with the commutator term)
Q.assert_gradient_error(rng, U + params_s, U + params_s, 1e-3, 1e-7)


# the preimage as nodes: inv accepts nodes, its backward follows from the
# Jacobian of the transport (g.ad.reverse.preimage); a gauge action of the
# preimage of two steps, w.r.t. the links and rho (the differences use the
# plain inverse)
grid4 = g.grid([4, 4, 4, 4], g.double)
rho_pre = g.complex(grid4)
rho_pre[:] = 0.07
U_pre = g.qcd.gauge.random(grid4, rng, scale=0.5)
even4, odd4 = g.even_odd_projectors(grid4)
description_pre = [(rho_pre, g.path().f(nu).f(0).b(nu).b(0)) for nu in range(1, 4)] + [
    (rho_pre, g.path().b(nu).f(0).f(nu).b(0)) for nu in range(1, 4)
]
steps_pre = [
    g.qcd.gauge.smear.directional_parallel_transport(U_pre, description_pre, 0, g(even4 + odd4), P1, [rho_pre])
    for P1 in [odd4, even4]
]
leaves_U = [rad.node(u) for u in U_pre]
leaf_rho = rad.node(rho_pre)
x_pre, x_plain = leaves_U, list(U_pre)
for phi in steps_pre:
    x_pre = phi.inv(x_pre + [leaf_rho])[0:4]
    x_plain = phi.inv(x_plain + [rho_pre])[0:4]
a_pre = g.qcd.gauge.action.iwasaki(5.5)
cost_pre = rad.functional_node(a_pre, x_pre).functional(*leaves_U, leaf_rho)
eps = abs(cost_pre(U_pre + [rho_pre]) - a_pre(x_plain)) / abs(a_pre(x_plain))
g.message(f"Preimage nodes: value vs plain inverse: {eps}")
assert eps < 1e-14
cost_pre.assert_gradient_error(rng, U_pre + [rho_pre], U_pre + [rho_pre], 1e-4, 1e-8)
cost_pre.assert_gradient_error(rng, U_pre + [rho_pre], [rho_pre], 1e-4, 1e-8)
