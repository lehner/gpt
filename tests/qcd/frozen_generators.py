#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2026
#
# Building blocks of frozen-shift generators (smear/frozen_generators.py):
# transverse smoothing, transporters of frozen links, the transverse Green's
# function, and generators built from them inside directional_frozen_shift
#
import gpt as g
import numpy as np

rad = g.ad.reverse
rng = g.random("frozen_generators")
smear = g.qcd.gauge.smear
fg = smear.frozen_generators

grid = g.grid([4, 4, 4, 4], g.double)
U = g.qcd.gauge.random(grid, rng, scale=0.5)
even, odd = g.even_odd_projectors(grid)
nd = 4


def relative(a, b):
    a, b = g.util.to_list(a), g.util.to_list(b)
    return (sum(g.norm2(x - y) for x, y in zip(a, b)) / sum(g.norm2(y) for y in b)) ** 0.5


def poisoned(U, mu):
    # U with nan in U_mu: a block that reads U_mu gives nan
    nan = g.lattice(U[mu])
    nan[:] = float("nan")
    return [nan if nu == mu else u for nu, u in enumerate(U)]


#####################################################################
# transverse staples and smoothing
#####################################################################
for mu in range(nd):
    dirs = [nu for nu in range(nd) if nu != mu]
    rho = np.array([[0.0 if (a == mu or b == mu or a == b) else 1.0 for b in range(nd)] for a in range(nd)])
    C_ref = g.qcd.gauge.staple_sum(U, rho=rho)
    C = fg.transverse_staples(mu)(U)
    eps = relative(C, [C_ref[nu] for nu in dirs])
    g.message(f"Transverse staples (mu = {mu}) vs staple_sum: {eps}")
    assert eps < 1e-14

    V = fg.transverse_stout(mu, 0.1, 3)(U)
    V_ref = U
    for _ in range(3):
        V_ref = smear.stout(rho=0.1, orthogonal_dimension=mu)(V_ref)
    eps = relative(V, V_ref)
    g.message(f"transverse_stout (mu = {mu}) vs stout(orthogonal_dimension): {eps}")
    assert eps < 1e-14 and V[mu] is U[mu]

    # never reads U_mu
    for name, block in [("stout", fg.transverse_stout(mu, 0.1, 2)), ("wilson flow", fg.transverse_wilson_flow(mu, 0.05, 2))]:
        W = block(poisoned(U, mu))
        n = sum(g.norm2(W[nu]) for nu in dirs)
        g.message(f"transverse {name} (mu = {mu}) with nan in U_mu: {n}")
        assert np.isfinite(n)

# the 4d Wilson flow (mu = None) against g.qcd.gauge.smear.wilson_flow, which
# integrates with the same scheme ((C.2) of arXiv:1006.4518, named
# runge_kutta_4 in g.algorithms.integrator): agreement to rounding checks the
# scheme and the normalization of Z
for epsilon in [0.04, 0.02]:
    W = fg.transverse_wilson_flow(None, epsilon, int(round(0.08 / epsilon)))(U)
    W_ref = U
    for _ in range(int(round(0.08 / epsilon))):
        W_ref = smear.wilson_flow(W_ref, epsilon)
    eps = relative(W, W_ref)
    change = relative(W, U)
    g.message(f"Wilson flow to t = 0.08 (epsilon = {epsilon}) vs wilson_flow: {eps}, change of the links {change}")
    assert eps < 1e-14 and change > 1e-2

# AD through the smoothing (a functional of the smoothed links)
a = g.lattice(U[0])
rng.element(a)
for name, block in [("stout", fg.transverse_stout(2, 0.1, 3)), ("wilson flow", fg.transverse_wilson_flow(2, 0.05, 2))]:
    aU = [rad.node(u) for u in U]
    W = block(aU)
    S = sum(g.inner_product(a, W[nu]).real for nu in [0, 1, 3]) + g.norm2(W[0] * W[1])
    g.message(f"Gradient through transverse {name}")
    S.functional(*aU).assert_gradient_error(rng, U, U, 1e-4, 1e-8)

#####################################################################
# frozen transporters
#####################################################################
mu, nu = 1, 3
# (with a closed loop: the plaquette in the transverse (nu, 0) plane)
T = fg.frozen_transporters(mu, even, fg.staple_paths(mu, nu) + [g.path().f(nu).f(0).b(nu).b(0)])
up, dn, plaq = T(U)
up_ref = U[nu] * g.cshift(U[mu], nu, 1) * g.adj(g.cshift(U[nu], mu, 1))
dn_ref = g.cshift(g.adj(U[nu]) * U[mu] * g.cshift(U[nu], mu, 1), nu, -1)
plaq_ref = U[nu] * g.cshift(U[0], nu, 1) * g.adj(g.cshift(U[nu], 0, 1)) * g.adj(U[0])
eps = relative([up, dn, plaq], [g(up_ref), g(dn_ref), g(plaq_ref)])
g.message(f"Frozen transporters vs cshift products: {eps}")
assert eps < 1e-14
for k, (P1, path) in enumerate(
    [
        (even, g.path().f(mu)),  # the updated link itself
        (even, g.path().f(nu).f(mu).b(nu).b(mu)),  # the plaquette through U_mu(x)
        (even, g.path().f(nu).f(nu).f(mu).b(nu).b(nu)),  # U_mu(x + 2 nu)
        (even, g.path().f(nu).f(mu)),  # ends at x + mu + nu
    ]
):
    try:
        fg.frozen_transporters(mu, P1, [path])
        assert False, "not rejected"
    except ValueError as e:
        g.message(f"Rejected: {e}")
# with a sparser mask (even sites with even x_nu) the path through
# U_mu(x + 2 nu) is frozen too
coor = g.coordinates(even)
sparse = g.complex(grid)
sparse[:] = 0
sparse[coor[coor[:, nu] % 4 == 0]] = 1
sparse = g(sparse * even)
fg.frozen_transporters(mu, sparse, [g.path().f(nu).f(nu).f(mu).b(nu).b(nu)])

#####################################################################
# transverse Green's function
#####################################################################
cg = g.algorithms.inverter.cg(eps=1e-14, maxiter=1000)
mu = 3
dirs = [nu for nu in range(nd) if nu != mu]
green = fg.transverse_green(mu, 0.5, cg)
s = g.lattice(grid, U[0].otype.cartesian())
rng.normal_element(s)
w = g.lattice(s)
rng.normal_element(w)
Aw = green.operator(*[U[nu] for nu in dirs], w)
Aw_ref = (0.5 + 2 * 3) * w
for nu in dirs:
    Aw_ref = Aw_ref - U[nu] * g.cshift(w, nu, 1) * g.adj(U[nu]) - g.cshift(g.adj(U[nu]) * w * U[nu], nu, -1)
eps = relative(Aw, g(Aw_ref))
g.message(f"Transverse Laplacian stencil vs cshift: {eps}")
assert eps < 1e-14
w = green(U, s)
eps = relative(green.operator(*[U[nu] for nu in dirs], w), s)
g.message(f"Transverse Green's function residual: {eps}, otype {w.otype.__name__}")
assert eps < 1e-12 and w.otype.__name__ == s.otype.__name__
aU = [rad.node(u) for u in U]
a_s = rad.node(s)
aw = green(aU, a_s)
a_alg = g.lattice(s)
rng.normal_element(a_alg)
S = g.inner_product(a_alg, aw).real + 0.3 * g.norm2(aw)
g.message("Gradient through the transverse Green's function")
# (w.r.t. the fields it reads: U_mu is not in the graph, and the functional
# returns no gradient for a leaf outside the graph)
S.functional(*aU, a_s).assert_gradient_error(rng, U + [s], [U[nu] for nu in dirs] + [s], 3e-4, 1e-8)

#####################################################################
# generators inside directional_frozen_shift
#####################################################################
V_gauge = g.lattice(U[0])
rng.element(V_gauge)


def green_generator(mu, P1, rho):
    # rho Green(P1 TA(S_+ S_-^dag)) (covariant: the Green's function maps an
    # adjoint field to an adjoint field).  The source must be restricted to P1:
    # off P1 the staples read updated links (U_mu(y +- nu) with y +- nu in
    # P1), and the non-local solve carries that dependence onto P1 (without
    # the restriction check_frozen raises)
    staples = fg.staple_pair_generator(mu, P1, (mu + 1) % nd, rho=1.0)
    G = fg.transverse_green(mu, 0.5, cg)

    def generator(F, xp):
        source = g.qcd.gauge.project.traceless_anti_hermitian(g(staples(F, xp) * P1))
        return g(rho * G(F, source))

    return generator


cases = [
    ("staple pair", 0, even, fg.staple_pair_generator(0, even, 2), [0.3]),
    ("smoothed staples (5 stout)", 1, odd, fg.smoothed_staple_generator(1, odd, fg.transverse_stout(1, 0.1, 5)), [0.3]),
    ("smoothed staples (2 RK3 Wilson flow)", 2, even, fg.smoothed_staple_generator(2, even, fg.transverse_wilson_flow(2, 0.05, 2)), [0.3]),
    ("Green's function", 3, odd, green_generator(3, odd, 2.0), []),
]
for name, mu, P1, G, params in cases:
    T = smear.directional_frozen_shift(U, mu, P1, G, parameters=params)
    Up = T(U + params)
    eps = relative(T.inv(Up)[0:4], U)
    change = relative(Up[0:4], U)
    T.check_frozen(U + params)
    cov = relative(T(g.qcd.gauge.transformed(U, V_gauge) + params)[0:4], g.qcd.gauge.transformed(Up[0:4], V_gauge))
    g.message(f"{name}: round trip {eps}, covariance {cov}, change of the links {change}")
    assert eps < 1e-14 and cov < 1e-13 and change > 1e-3
    indices = list(range(4 + len(params)))
    a = g.qcd.gauge.action.iwasaki(5.5).transformed(T, indices=indices, projection=[0, 1, 2, 3])
    g.message(f"{name}: force")
    a.assert_gradient_error(rng, U + params, U + params, 1e-4, 1e-8)

# the Green's function of the unrestricted source reads updated links through
# the non-local solve: detected
staples = fg.staple_pair_generator(3, odd, 0, rho=1.0)
G = fg.transverse_green(3, 0.5, cg)
T = smear.directional_frozen_shift(
    U, 3, odd, lambda F, xp: G(F, g.qcd.gauge.project.traceless_anti_hermitian(staples(F, xp)))
)
try:
    T.check_frozen(U)
    assert False, "not detected"
except ValueError as e:
    g.message(f"Detected: {e}")
