#!/usr/bin/env python3
#
# The spectral single-loop step (g.qcd.gauge.smear.spectral_loop): analytic
# log det against the 8x8 Jacobian of the transport, inverse, forces and
# coefficient gradients, second derivatives of the latent action, and
# coefficients computed by an upstream node graph
#
import gpt as g
import numpy as np

rad = g.ad.reverse
rng = g.random("spectral_loop")
grid = g.grid([4, 4, 4, 8], g.double)
U = g.qcd.gauge.random(grid, rng, scale=0.5)
even, odd = g.even_odd_projectors(grid)
for name in ["fixed_point"]:
    g.default.set_verbose(name, False)


def site_field(c0, scale):
    # a site-dependent complex coefficient field around c0
    c = g.complex(grid)
    rng.cnormal(c)
    return g(c0 * even + c0 * odd + scale * c)


def rel(a, b):
    return sum(g.norm2(x - y) for x, y in zip(a, b)) / sum(g.norm2(y) for y in b)


cases = [
    ("fixed {1}", {1: 0.12}, []),
    ("fixed {1,2,3}", {1: 0.12, 2: 0.05, 3: 0.02}, []),
    ("numbers {1,2}", {1, 2}, [np.array(0.12 + 0.03j), np.array(0.05 - 0.02j)]),
    ("fields {1,3}", {1, 3}, [site_field(0.12, 0.02), site_field(0.02 + 0.01j, 0.005)]),
]

for name, coefficients, cs in cases:
    for cb, nu0, forward in [(g.even, 1, True), (g.odd, 2, False)]:
        tag = f"{name}, {'even' if cb is g.even else 'odd'}, nu0 {nu0}, {'forward' if forward else 'backward'}"
        step = g.qcd.gauge.smear.spectral_loop(
            dimension=0, checkerboard=cb, nu0=nu0, forward=forward, coefficients=coefficients
        )
        fields = U + cs
        Up = step(fields)

        # the analytic log det against that of the 8x8 Jacobian of the
        # transport (same loop and loop function)
        t = step.transport(fields)[0]
        ld = step.log_det_jacobian(fields)
        ld_ref = g(g.component.real(t.log_det_jacobian_field(fields)))
        eps2 = g.norm2(ld - ld_ref) / g.norm2(ld_ref)
        g.message(f"{tag}: log det vs 8x8 Jacobian: {eps2}")
        assert eps2 < 1e-24

        # inverse
        eps2 = rel(step.inv(Up)[0:4], U)
        g.message(f"{tag}: inverse: {eps2}")
        assert eps2 < 1e-26

        # forces and coefficient gradients of a latent action
        W = g.qcd.gauge.action.iwasaki(2.95).transformed(
            step, indices=list(range(len(fields))), projection=[0, 1, 2, 3]
        ) + step.action_log_det_jacobian()
        W.assert_gradient_error(rng, fields, fields, 1e-4, 1e-8)

        # the same latent action fused (one reverse pass for the pull-back and
        # the log det): values and gradients (all fields, links only, the
        # coefficients only) against the plain chain
        Wf = step.fused_action(
            g.qcd.gauge.action.iwasaki(2.95), indices=list(range(len(fields))), projection=[0, 1, 2, 3]
        )
        eps = abs(Wf(fields) - W(fields)) / abs(W(fields))
        for dfields in [fields, fields[0:4], fields[4:]]:
            if dfields:
                eps2 = rel(Wf.gradient(fields, dfields), W.gradient(fields, dfields))
                g.message(f"{tag}: fused vs plain chain, {len(dfields)} fields: value {eps}, gradient {eps2}")
                assert eps < 1e-14 and eps2 < 1e-26

# a chain of steps (spectral steps fused, a local_stout step as transformed
# plus its log det) against the plain chain
steps = [
    g.qcd.gauge.smear.spectral_loop(dimension=0, checkerboard=g.even, nu0=1, coefficients={1: 0.12, 2: 0.03}),
    g.qcd.gauge.smear.local_stout(dimension=1, checkerboard=g.odd, rho=0.1),
    g.qcd.gauge.smear.spectral_loop(dimension=3, checkerboard=g.odd, nu0=2, forward=False, coefficients={1: 0.1}),
]
S = g.qcd.gauge.action.iwasaki(2.95)
Wf = g.qcd.gauge.smear.latent_action(S, steps)
W = S
for s in steps:
    W = W.transformed(s) + s.action_log_det_jacobian()
eps = abs(Wf(U) - W(U)) / abs(W(U))
eps2 = rel(Wf.gradient(U, U), W.gradient(U, U))
g.message(f"latent_action chain vs plain chain: value {eps}, gradient {eps2}")
assert eps < 1e-14 and eps2 < 1e-26
Wf.assert_gradient_error(rng, U, U, 1e-4, 1e-8)

# learnable coefficients equal to fixed ones give the same map and log det
fixed = g.qcd.gauge.smear.spectral_loop(dimension=1, checkerboard=g.odd, nu0=3, coefficients={1: 0.1, 2: 0.04})
learn = g.qcd.gauge.smear.spectral_loop(dimension=1, checkerboard=g.odd, nu0=3, coefficients={1, 2})
cs = [np.array(0.1 + 0j), np.array(0.04 + 0j)]
eps2 = rel(learn(U + cs)[0:4], fixed(U))
eps2 += g.norm2(learn.log_det_jacobian(U + cs) - fixed.log_det_jacobian(U))
g.message(f"learnable vs fixed: {eps2}")
assert eps2 < 1e-28

# the bound sum_k k |c_k| < 1: fixed coefficients at construction, learnable
# ones on plain values (here one site of P1 with a large c_1)
try:
    g.qcd.gauge.smear.spectral_loop(dimension=0, checkerboard=g.even, nu0=1, coefficients={1: 0.5, 2: 0.3})
    assert False
except ValueError as e:
    g.message(f"rejected: {e}")
c1 = site_field(0.1, 0.0)
c1[0, 0, 0, 0] = 0.95
step = g.qcd.gauge.smear.spectral_loop(dimension=0, checkerboard=g.even, nu0=1, coefficients={1, 2})
for f in [step, step.inv]:
    try:
        f(U + [c1, np.array(0.05 + 0j)])
        assert False
    except ValueError as e:
        g.message(f"rejected: {e}")

# second derivatives of the latent action W(V) = S(step(V)) - sum log det,
# as a node graph (recorded pass): the Hessian bilinear against a second
# difference of W, and the mixed derivative d/dc_k <dA, dW/dU> against a
# difference of the force
action = g.qcd.gauge.action.differentiable_iwasaki(2.95)
step = g.qcd.gauge.smear.spectral_loop(dimension=2, checkerboard=g.even, nu0=0, forward=False, coefficients={1, 2})
cs = [site_field(0.12, 0.02), site_field(0.04, 0.01)]
fields = U + cs


def latent(x):
    return action(step(x)[0:4]) - g.component.real(g.sum(step.log_det_jacobian(x)))


def latent_value(x):
    return latent([rad.node(v, with_gradient=False) for v in x])(with_gradients=False).real


dA = rng.normal_element(g.group.cartesian(U))
dc = [rng.cnormal(g.complex(grid)) for _ in cs]
nodes = [rad.node(v) for v in fields]
latent(nodes).backward(create_graph=True)
dW = [x.gradient for x in nodes]
sum(g.group.inner_product(dA[mu], dW[mu]) for mu in range(4)).backward()
H = [x.gradient for x in nodes]
dA_H_dA = sum(g.group.inner_product(dA[mu], H[mu]) for mu in range(4))


def along(t):
    return [g(g.group.compose(g(t * dA[mu]), U[mu])) for mu in range(4)] + cs


h = 1e-3
fd = (latent_value(along(h)) - 2 * latent_value(along(0.0)) + latent_value(along(-h))) / h**2
err = abs(dA_H_dA - fd) / abs(fd)
g.message(f"latent action: Hessian bilinear {dA_H_dA} vs second difference {fd}: {err}")
assert err < 1e-5

# d/dc <dA, dW/dU> along dc against a difference of the force (the
# functional of the first-order graph)
mixed = sum(g.group.inner_product(dc[i], H[4 + i]) for i in range(len(cs)))
nodes1 = [rad.node(v) for v in fields]
force = latent(nodes1).functional(*nodes1)


def force_along(t):
    fs = U + [g(c + t * d) for c, d in zip(cs, dc)]
    F = force.gradient(fs, fs[0:4])
    return sum(g.group.inner_product(dA[mu], F[mu]) for mu in range(4))


fd = (force_along(h) - force_along(-h)) / (2 * h)
err = abs(mixed - fd) / abs(fd)
g.message(f"latent action: d/dc <dA, dW/dU> {mixed} vs difference of the force {fd}: {err}")
assert err < 1e-5

# coefficients computed by an upstream node graph: c_1 = 0.1 Re w, c_2 =
# 0.05 w^2 (w a complex field), gradient w.r.t. the links and w
w = site_field(1.0, 0.1)
leaves = [rad.node(v) for v in U + [w]]
c = [g.component.real(leaves[4]) * 0.1, leaves[4] * leaves[4] * 0.05]
S = latent(leaves[0:4] + c)
S.functional(*leaves).assert_gradient_error(rng, U + [w], U + [w], 1e-4, 1e-8)

# the transports are built by their first call, here with nodes: the
# preimages of a latent chain with coefficient leaves.  The cached state must
# not capture these (formerly the identity of the log det became a node of
# that graph: the force of the latent action, evaluated inside the outer
# reverse pass, reset its flows).  Loss: L(c) = Re sum_x [sum_mu tr(F_mu w_mu)
# + tr V_0], F = grad_V W(V, c) (gradient_finite_difference), V = latent(U, c)
nd = 4
cvals = [0.10, 0.03, 0.01]
dirs = [1.0, -0.7, 0.4]
one = g.complex(grid)
one[:] = 1.0
cfields = [g(v * one) for v in cvals]
w = [rng.normal_element(g.group.cartesian(u)) for u in U]


def latent_chain():
    steps = [
        g.qcd.gauge.smear.spectral_loop(0, g.even, 1, True, coefficients={1, 2, 3}),
        g.qcd.gauge.smear.spectral_loop(1, g.odd, 2, False, coefficients={1, 2, 3}),
    ]
    W = g.qcd.gauge.action.wilson(5.5)
    for k, s in enumerate(steps):
        W = s.fused_action(W, projection=list(range(nd)) if k == 0 else None)
    return steps, W


def preimage(steps, cs):
    x = list(U)
    for s in steps:
        x = s.inv(x + list(cs), max_iter=200)[0:nd]
    return x


def loss_plain(cs):
    steps, W = latent_chain()
    V = preimage(steps, cs)
    F = W.gradient(V + cs, V)
    return sum(g.sum(g.trace(F[mu] * w[mu])).real for mu in range(nd)) + g.sum(g.trace(V[0])).real


steps, W = latent_chain()
r = [rad.node(g.copy(c)) for c in cfields]
V = preimage(steps, r)
F = rad.symplectic.gradient_finite_difference(W, nd)(V + r)
wn = [rad.node(x, with_gradient=False) for x in w]
L = None
for mu in range(nd):
    t = g.component.real(g.sum(g.trace(F[mu] * wn[mu])))
    L = t if L is None else L + t
L = L + g.component.real(g.sum(g.trace(V[0])))
L.backward()
ad = sum(d * g.sum(x.gradient).real for d, x in zip(dirs, r))
h = 1e-5
fd = (
    loss_plain([g(c + (h * d) * one) for c, d in zip(cfields, dirs)])
    - loss_plain([g(c - (h * d) * one) for c, d in zip(cfields, dirs)])
) / (2 * h)
err = abs(ad / fd - 1)
g.message(f"transports built from nodes: dL/dc along dc {ad} vs central difference {fd}: {err}")
assert err < 1e-6
for s in steps:
    t = s.cache[next(iter(s.cache))][0]
    assert not any(isinstance(p, rad.node_base) for p in t.parameters)
Vp = [rad.util.value_of(v) if isinstance(v, rad.node_base) else v for v in V]
assert not isinstance(W(Vp + cfields), rad.node_base)

# directional_parallel_transport rejects node parameters (templates only)
try:
    g.qcd.gauge.smear.directional_parallel_transport(
        U, [(1.0, g.path().f(1).f(0).b(1).b(0))], 0, None, even, [rad.node(cfields[0])]
    )
    assert False
except ValueError as e:
    g.message(f"rejected: {e}")
