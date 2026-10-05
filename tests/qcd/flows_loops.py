#!/usr/bin/env python3
#
# directional_parallel_transport with fixed loops as further inputs of the
# loop function: closed loops at x that avoid every link the step updates
# (U_0 on the sites of P1), here the 2x1 rectangles around U_0(x) (the two
# plaquettes sharing it, U_0(x) cancels) and a plaquette in a plane without
# direction 0; f(sm, xparams, L) with the fixed loops L enters the local
# paths like the staple.  (Separate from flows.py, which uses nearly all the
# memory of the test machine under MPI.)
#
import gpt as g

rng = g.random("test")
grid4 = g.grid([4, 4, 4, 8], g.double)


def loop_function_p(sm, xparams):
    return sm + sm * sm * xparams[1]


# -sum_x w(x) log det J(x) with a prescribed staple (and loops)
class weighted_log_det(g.group.differentiable_functional):
    def __init__(self, t, w, staple, loops=None):
        self.t, self.w, self.staple, self.loops = t, w, staple, loops

    def __call__(self, fields):
        ld = self.t.log_det_jacobian_field(fields, self.staple, self.loops)
        return -complex(g.sum(g(self.w * g.component.real(ld)))).real

    def gradient(self, fields, dfields):
        return self.t.weighted_log_det_jacobian_gradient(
            fields, dfields, self.w, self.staple, self.loops
        )


loops_fixed = [g.path().f(nu).f(0).b(nu).b(nu).b(0).f(nu) for nu in range(1, 4)] + [
    g.path().f(1).f(2).b(1).b(2)
]


def loop_function_loops(sm, xparams, loops):
    f = sm
    for L in loops:
        q = g.component.real(g(g.trace(L) * (1.0 / 3.0)))
        f = g(f + sm * sm * q * xparams[1] + sm * L * 0.05)
    return f


U_l = g.qcd.gauge.random(grid4, rng, scale=0.5)
even4, odd4 = g.even_odd_projectors(grid4)
full4 = g(even4 + odd4)
rho_l = g.complex(grid4)
rho_l[:] = 0.05
description_l = [(rho_l, g.path().f(nu).f(0).b(nu).b(0)) for nu in range(1, 4)] + [
    (rho_l, g.path().b(nu).f(0).f(nu).b(0)) for nu in range(1, 4)
]
p_l = g.complex(grid4)
rng.normal(p_l)
p_l[:] = 0.1 + 0.02 * p_l[:]
params_l = [rho_l, p_l]


def make_pt_loops(P1, generic=False, params=params_l, loops=loops_fixed):
    pt = g.qcd.gauge.smear.directional_parallel_transport(
        U_l, description_l, 0, full4, P1, params, loop_function=loop_function_loops, loops=loops
    )
    if generic:
        pt.description_staple = None
    return pt


pt_l = make_pt_loops(odd4)
pt_l_generic = make_pt_loops(odd4, generic=True)

# the loops are the plain transported loops, and they change the flow
for k, path in enumerate(loops_fixed):
    ref = g.parallel_transport_matrix(U_l, [(0, -1, 1.0, path)], 1)(U_l)
    eps2 = g.norm2(pt_l._loops(U_l + params_l)[k] - ref) / g.norm2(ref)
    assert eps2 < 1e-28
# P0 (a factor of U_0, here a real field 0.8..1.2) applies to the loops as to
# the staple: the loops of the masked links, local and generic paths agree
P0_l = g.complex(grid4)
rng.uniform_real(P0_l, min=0.8, max=1.2)
pt_P0 = g.qcd.gauge.smear.directional_parallel_transport(
    U_l,
    description_l,
    0,
    P0_l,
    odd4,
    params_l,
    loop_function=loop_function_loops,
    loops=loops_fixed,
)
pt_P0_generic = g.qcd.gauge.smear.directional_parallel_transport(
    U_l,
    description_l,
    0,
    P0_l,
    odd4,
    params_l,
    loop_function=loop_function_loops,
    loops=loops_fixed,
)
pt_P0_generic.description_staple = None
U_masked = [g(U_l[0] * P0_l)] + U_l[1:]
for k, path in enumerate(loops_fixed):
    ref = g.parallel_transport_matrix(U_masked, [(0, -1, 1.0, path)], 1)(U_masked)
    eps2 = g.norm2(pt_P0._loops(U_l + params_l)[k] - ref) / g.norm2(ref)
    assert eps2 < 1e-28
fields_P0 = U_l + params_l
gr0 = pt_P0.action_log_det_jacobian().gradient(fields_P0, fields_P0)
gr1 = pt_P0_generic.action_log_det_jacobian().gradient(fields_P0, fields_P0)
eps2 = sum(g.norm2(x - y) for x, y in zip(gr0, gr1)) / sum(g.norm2(x) for x in gr0)
g.message(f"Fixed loops: P0: local vs generic log-det force: {eps2}")
assert eps2 < 1e-20

Uprime0 = g.qcd.gauge.smear.directional_parallel_transport(
    U_l, description_l, 0, full4, odd4, params_l, loop_function=loop_function_p
)(U_l + params_l)[0]
Uprime1 = pt_l(U_l + params_l)[0]
eps2 = g.norm2(Uprime0 - Uprime1) / g.norm2(Uprime0)
g.message(f"Fixed loops change the flow: {eps2}")
assert eps2 > 1e-8

# gauge covariance of the step
V = g.mcolor(U_l[0].grid)
rng.element(V)
UV = g.qcd.gauge.transformed(U_l, V)
Uprime1V = pt_l(UV + params_l)[0]
Uprime1_V = g.qcd.gauge.transformed(pt_l(U_l + params_l)[0:4], V)[0]
eps2 = g.norm2(Uprime1V - Uprime1_V) / g.norm2(Uprime1V)
g.message(f"Fixed loops: covariance: {eps2}")
assert eps2 < 1e-28

# local and generic paths agree: forward, Jacobian block, log det, Jacobian
# and log-det force (links and parameters)
fields_l = U_l + params_l
Up0, Up1 = pt_l(fields_l), pt_l_generic(fields_l)
eps2 = sum(g.norm2(x - y) for x, y in zip(Up0[0:4], Up1[0:4])) / g.norm2(Up0[0])
g.message(f"Fixed loops: local vs generic forward: {eps2}")
assert eps2 < 1e-28
M0 = pt_l.jacobian_matrix(fields_l)
M1 = pt_l_generic.jacobian_matrix(fields_l)
eps2 = g.norm2(M0 - M1) / g.norm2(M0)
g.message(f"Fixed loops: local vs generic Jacobian block: {eps2}")
assert eps2 < 1e-25
v0, v1 = pt_l.log_det_jacobian(fields_l), pt_l_generic.log_det_jacobian(fields_l)
eps = abs(v0 - v1) / abs(v0)
g.message(f"Fixed loops: local vs generic log-det: {eps}")
assert eps < 1e-12
dirs = [rng.normal_element(g.group.cartesian(x)) for x in fields_l]
for name, gr0, gr1 in [
    (
        "log-det force",
        pt_l.action_log_det_jacobian().gradient(fields_l, fields_l),
        pt_l_generic.action_log_det_jacobian().gradient(fields_l, fields_l),
    ),
    (
        "Jacobian",
        pt_l.jacobian(fields_l, Up0, dirs),
        pt_l_generic.jacobian(fields_l, Up0, dirs),
    ),
]:
    for x, y in zip(gr0, gr1):
        eps2 = g.norm2(x - y) / g.norm2(x)
        g.message(f"Fixed loops: local vs generic {name}: {eps2}")
        assert eps2 < 1e-20

# gradients w.r.t. links, rho and p (local and generic)
for generic in [False, True]:
    a1 = g.qcd.gauge.action.iwasaki(6)
    a1 = a1.transformed(
        make_pt_loops(odd4, generic), indices=[0, 1, 2, 3, 4, 5], projection=[0, 1, 2, 3]
    )
    a1 = a1.transformed(
        make_pt_loops(even4, generic), indices=[0, 1, 2, 3, 4, 5], projection=[0, 1, 2, 3, 4, 5]
    )
    a1.assert_gradient_error(rng, fields_l, fields_l, 1e-4, 1e-7)
    a2 = (pt_l_generic if generic else pt_l).action_log_det_jacobian()
    a2.assert_gradient_error(rng, fields_l, fields_l, 1e-4, 1e-7)
    a2.assert_gradient_error(rng, fields_l, [p_l], 3e-3, 1e-7)

# inverse (fixed point, and with Newton steps only)
for newton_rate in [0.5, 0.0]:
    Uinv = pt_l.inv(Up0[0:4] + params_l, newton_rate=newton_rate)
    eps2 = g.norm2(Uinv[0] - U_l[0]) / g.norm2(U_l[0])
    g.message(f"Fixed loops: inverse ({pt_l.inverse_newton_iterations} Newton iterations): {eps2}")
    assert eps2 < 1e-25
    assert newton_rate > 0 or pt_l.inverse_newton_iterations > 0

# a loop through an updated link is rejected: the plaquette through U_0(x),
# and the rectangle through U_0(x + 2 e_1) (same checkerboard as x)
for p_bad in [g.path().f(0).f(1).b(0).b(1), g.path().f(1, 2).f(0).b(1, 2).b(0)]:
    try:
        make_pt_loops(odd4, loops=[p_bad])
        assert False
    except ValueError as e:
        g.message(f"Fixed loops: rejected: {e}")
# ... and accepted with a full4-lattice P1 only if it avoids U_0 entirely
make_pt_loops(full4, loops=[g.path().f(1).f(2).b(1).b(2)])
try:
    make_pt_loops(full4, loops=loops_fixed[0:1])
    assert False
except ValueError as e:
    g.message(f"Fixed loops: rejected for P1 = 1: {e}")

# a single loop (a single stencil output): local vs generic log-det force
pt_1, pt_1_generic = [
    make_pt_loops(odd4, generic, loops=loops_fixed[1:2]) for generic in [False, True]
]
gr0 = pt_1.action_log_det_jacobian().gradient(fields_l, fields_l)
gr1 = pt_1_generic.action_log_det_jacobian().gradient(fields_l, fields_l)
eps2 = sum(g.norm2(x - y) for x, y in zip(gr0, gr1)) / sum(g.norm2(x) for x in gr0)
g.message(f"Fixed loops: one loop: local vs generic log-det force: {eps2}")
assert eps2 < 1e-20

# the numerical check of inv: a loop through the updated links (here
# bypassing the check of the constructor) changes in the update
pt_1.loops = [g.path().f(0).f(1).b(0).b(1)]
pt_1._loop_cache = {}
message = None
try:
    pt_1.inv(pt_1(fields_l)[0:4] + params_l)
except RuntimeError as e:
    message = str(e)
g.message(f"Fixed loops: inverse: {message}")
assert message is not None and "depends on the updated links" in message

# the weighted log det with a prescribed staple, and with prescribed loops
w_l = g.complex(U_l[0].grid)
rng.uniform_real(w_l)
W_l = [g.mcolor(U_l[0].grid) for _ in range(6)]
rng.element(W_l, scale=3.0)
staple_l = lambda fs: sum(fs[4] * W for W in W_l)
# (prescribed loops may depend on any field, here also on U_0 and p)
loops_l = lambda fs: [g(fs[0] * g.adj(fs[nu]) * fs[5]) for nu in range(1, 4)] + [g(fs[1] * fs[2])]
for loops_arg in [None, loops_l]:
    lw = weighted_log_det(pt_l, w_l, staple_l, loops_arg)
    lw.assert_gradient_error(rng, fields_l, fields_l, 1e-4, 1e-8)
