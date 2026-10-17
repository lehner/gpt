#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2026
#
# Frozen-generator shifts in field-transformed HMC, as in the FTHMC
# experiment scripts: a chain of frozen_shift / directional_frozen_shift and
# local_stout steps, the latent field by successive inverses, the latent action
# S o Phi - log det, molecular dynamics (Delta H scaling, reversibility), and a
# transformed pseudofermion action
#
import gpt as g
import numpy as np

rng = g.random("flows_frozen_hmc")
smear = g.qcd.gauge.smear
fg = smear.frozen_generators
g.default.set_verbose("cg", False)

grid = g.grid([4, 4, 4, 8], g.double)
U0 = g.qcd.gauge.random(grid, rng, scale=0.3)
even, odd = g.even_odd_projectors(grid)

coor = g.coordinates(even)
quarter = g.complex(grid)
quarter[:] = 0
quarter[coor[coor[:, 3] % 2 == 0]] = 1
quarter = g(quarter * even)


def relative(a, b):
    return (sum(g.norm2(x - y) for x, y in zip(a, b)) / sum(g.norm2(y) for y in b)) ** 0.5


# Phi = sm[0] o sm[1] o ... (sm[-1] acts first on the latent field)
sm = [
    smear.frozen_shift(dimension=0, checkerboard=g.even, generator=fg.staple_pair_generator(0, even, 2, rho=0.3)),
    smear.local_stout(rho=0.05, dimension=1, checkerboard=g.odd),
    smear.frozen_shift(
        dimension=1,
        checkerboard=g.odd,
        generator=fg.smoothed_staple_generator(1, odd, fg.transverse_stout(1, 0.1, 2), rho=0.2),
    ),
    smear.directional_frozen_shift(U0, 3, quarter, fg.staple_pair_generator(3, quarter, 0, rho=0.3)),
    smear.local_stout(rho=0.05, dimension=2, checkerboard=g.even),
]

# the latent field of U0
U = U0
for s in sm:
    U = s.inv(U)
Up = U
for s in reversed(sm):
    Up = s(Up)
eps = relative(Up, U0)
g.message(f"Phi(latent) = U0: {eps}")
assert eps < 1e-12

# the latent action S o Phi - log det (one chain each, as fthmc-learn.py)
a_gauge = g.qcd.gauge.action.wilson(5.5)
S = a_gauge
ld = None
for s in sm:
    S = S.transformed(s)
    ld = s.action_log_det_jacobian() if ld is None else ld.transformed(s) + s.action_log_det_jacobian()
S = S + ld
eps = abs(S(U) - a_gauge(U0) - ld(U)) / abs(a_gauge(U0))
g.message(f"Latent action vs physical action - log det: {eps}")
assert eps < 1e-12
g.message("Latent force")
S.assert_gradient_error(rng, U, U, 1e-4, 1e-8)

# molecular dynamics on the latent field
a0 = g.qcd.scalar.action.mass_term()
p = g.group.cartesian(U)
a0.draw(p, rng)
U_start, p_start = g.copy(U), g.copy(p)
Q = g.copy(U)
sympl = g.algorithms.integrator.symplectic
ip = sympl.update_p(p, lambda: S.gradient(Q, Q))
iq = sympl.update_q(Q, lambda: a0.gradient(p, p))


def H():
    return a0(p) + S(Q)


def reset():
    g.copy(Q, U_start)
    g.copy(p, p_start)


tau = 0.5
# the order of the integrators from the differences of the final links for n,
# 2n and 4n steps (|Q_n - Q_2n| / |Q_2n - Q_4n| = 2^order; a field norm,
# robust), and the signed Delta H.  The |Delta H| of a single trajectory is a
# robust order test only where its leading term dominates (leapfrog here); for
# OMF4 its dt^4 coefficient can be accidentally small on a trajectory (seen on
# 4^3 x 8: Delta H changes sign between 4 and 8 steps)
for name, integrator, order, n0 in [("leap_frog", sympl.leap_frog, 2, 4), ("OMF4", sympl.OMF4, 4, 4)]:
    dH, Qn = [], []
    for n in [n0, 2 * n0, 4 * n0]:
        reset()
        h0 = H()
        integrator(n, ip, iq)(tau)
        dH.append(H() - h0)
        Qn.append(g.copy(Q))
    d = [relative(Qn[0], Qn[1]), relative(Qn[1], Qn[2])]
    ratio = d[0] / d[1]
    g.message(f"{name}: Delta H = {dH}, link differences {d} (ratio {ratio}, expected {2 ** order})")
    assert abs(np.log2(ratio) - order) < 0.4
    if order == 2:
        assert all(abs(np.log2(abs(dH[i] / dH[i + 1])) - order) < 0.4 for i in range(2))

    # reversibility: forward, flip the momenta, forward, flip
    reset()
    I = integrator(4, ip, iq)
    I(tau)
    for x in p:
        x @= -x
    I(tau)
    for x in p:
        x @= -x
    eps_U = relative(Q, U_start)
    eps_p = relative(p, p_start)
    g.message(f"{name}: reversibility {eps_U} (links), {eps_p} (momenta)")
    assert eps_U < 1e-12 and eps_p < 1e-10

# a pseudofermion action (two-flavor Wilson) transformed by a frozen step and a
# local_stout step as in the production scripts (applications/hmc/dwf/
# ensemblesNPR.py, transform): the steps act on the links, draw applies them
# to the links and appends the pseudofermion
U = g.qcd.gauge.random(grid, rng, scale=0.3)
M = g.qcd.fermion.wilson_clover(
    U,
    {
        "kappa": 0.12,
        "csw_r": 0.0,
        "csw_t": 0.0,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)
sol = g.algorithms.inverter.cg({"eps": 1e-12, "maxiter": 1024})
psi = g.vspincolor(M.F_grid)
fs = smear.frozen_shift(
    dimension=2,
    checkerboard=g.odd,
    generator=fg.smoothed_staple_generator(2, odd, fg.transverse_wilson_flow(2, 0.05, 1), rho=0.3),
)
ls = smear.local_stout(rho=0.05, dimension=0, checkerboard=g.even)


def transform(a, s):
    a_transformed = a.transformed(s, indices=list(range(len(U))))

    def draw(fields, rng, *extra):
        return a.draw(s(fields[0 : len(U)]) + fields[len(U) :], rng, *extra)

    a_transformed.draw = draw
    return a_transformed


a_pf = g.qcd.pseudofermion.action.two_flavor(M, sol)
a_t = a_pf
for s in [fs, ls]:  # f(U) -> f(fs(U)) -> f(fs(ls(U)))
    a_t = transform(a_t, s)
fields = U + [psi]
value_draw = a_t.draw(fields, rng)
value = a_t(fields)
eps = abs(value_draw - value) / abs(value)
g.message(f"Transformed pseudofermion action: drawn {value_draw} vs computed {value} ({eps})")
assert eps < 1e-8

# the drawn pseudofermion is that of the physical links fs(ls(U))
value_physical = a_pf(fs(ls(U)) + [psi])
eps = abs(value_physical - value) / abs(value)
g.message(f"Transformed pseudofermion action vs action of the physical links: {eps}")
assert eps < 1e-12

g.message("Transformed pseudofermion force")
a_t.assert_gradient_error(rng, fields, U, 1e-3, 1e-7)
