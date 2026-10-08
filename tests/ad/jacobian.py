#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2026
#
# g.ad.reverse.jacobian: the site-diagonal Jacobian of a node graph as a
# node (forward tangents replayed on the graph), against the rows of
# reverse-mode VJPs; the force of -sum log det J (one reverse pass) against
# finite differences; chunked replays; the per-site matrix functions;
# site-locality errors; dpt's local map against dpt's own block and force.
#
import gpt as g
import numpy as np

rad = g.ad.reverse
grid = g.grid([4, 4, 4, 4], g.double)
rng = g.random("jacobian")
U = g.qcd.gauge.random(grid, rng, scale=0.5)
x0 = U[0]
# (a staple-like constant, scaled as by rho: the site maps stay invertible)
C0 = g(0.1 * (U[1] * U[2] + U[3]))
T = g.group.algebra_kernels(grid, x0.otype.cartesian()).fgenerators
kernels = g.group.algebra_kernels(grid, x0.otype.cartesian())
ng = len(T)
P = g.copy(x0)

# site maps x -> y = exp(TA(f(C x^dag))) x with f of different kinds (each
# with its parameters as further leaves)
poly = g.ml.layer.polynomial(P, 3)
poly.initialize(rng, 0.2)
inv_layer = g.ml.layer.matrix_invariants(P)
mlp = g.ml.layer.mlp(P, inv_layer.n, g.ml.layer.matrix_words.n, width=4, depth=1)
words = g.ml.layer.matrix_words(P)
(s,) = g.ml.symbols("P")
(I,) = inv_layer([s], name="invariants")
(c,) = mlp([I], name="mlp")
(f,) = words([s, c], name="words")
net = g.ml.pack(f=f).function()
net.initialize(rng, 0.1)
net.calibrate([[g(C0 * g.adj(x0))]])
coefficient = np.array(0.3 + 0.1j)

models = {
    "none": ([], lambda p, xp: p),
    "arithmetic": ([coefficient], lambda p, xp: p + xp[0][()] * p * g.adj(p) * p + p**3 * 0.05),
    "polynomial": (list(poly.parameters()), lambda p, xp: poly([p], xp)[0]),
    "mlp": (list(net.parameters()), lambda p, xp: net([p], xp)[0]),
}


def rel(a, b):
    # |a - b|^2 / |b|^2 of fields, numbers or arrays
    if isinstance(b, g.lattice):
        return g.norm2(a - b) / g.norm2(b)
    a, b = np.asarray(a, dtype=np.complex128), np.asarray(b, dtype=np.complex128)
    return float(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2))


def local_map(x, C, xp, fn):
    return g.matrix.exp(g.qcd.gauge.project.traceless_anti_hermitian(fn(C * g.adj(x), xp))) * x


def reference(fields, fn):
    # J[b, a] from VJPs: the cartesian gradient of the output seeded with
    # the generator T_b has the coordinates J[b, a] (orthonormal generators)
    nodes = [rad.node(v) for v in fields]
    y = local_map(nodes[0], nodes[1], nodes[2:], fn)
    yv = y(with_gradients=False, retain_values=True)
    rows = []
    for Tb in T:
        nodes[0].gradient = None
        y.backward(initial_gradient=g.cartesian_to_infinitesimal(yv, Tb), retain_values=True, wrt=[nodes[0]])
        rows.append(nodes[0].gradient)
    M = g.lattice(grid, x0.otype.jacobian_otype())
    kernels.rows(M, rows)
    return M  # M[b, a] = J[b, a]


def jacobian_node(nodes, fn, chunk=None):
    x = rad.identity(nodes[0])
    return rad.jacobian(local_map(x, nodes[1], nodes[2:], fn), x, chunk)


for name, (params, fn) in models.items():
    fields = [x0, C0] + params
    nodes = [rad.node(v) for v in fields]
    J = jacobian_node(nodes, fn)(with_gradients=False)
    M = reference(fields, fn)
    err = g.norm2(J - M) / g.norm2(M)
    g.message(f"{name}: J vs VJP rows: {err:.2e}")
    assert err < 1e-26

    # the force of S = -sum Re log det J (all leaves) against differences;
    # C and the parameters are additive
    nodes = [rad.node(v) for v in fields]
    S = g.component.real(g.sum(rad.log_det(jacobian_node(nodes, fn)))) * (-1.0)
    fS = S.functional(*nodes)
    fS.assert_gradient_error(rng, fields, fields, 1e-4, 1e-8)

    # chunked replays: the same J and force
    nodes_c = [rad.node(v) for v in fields]
    Sc = g.component.real(g.sum(rad.log_det(jacobian_node(nodes_c, fn, chunk=3)))) * (-1.0)
    fSc = Sc.functional(*nodes_c)
    ga, gb = fS.gradient(fields, fields), fSc.gradient(fields, fields)
    err = max(rel(a, b) for a, b in zip(ga, gb))
    g.message(f"{name}: chunk 3 vs unchunked force: {err:.2e}, values {abs(fS(fields) - fSc(fields)):.2e}")
    assert err < 1e-26

# per-site matrix functions on a general complex matrix field: value, force
# and a second derivative (recorded pass) against differences
A = g.mcomplex(grid, 4)
rng.cnormal(A)
A @= A + g.identity(A) * 4.0
for name, fn in [
    ("log_det", lambda a: g.sum(rad.log_det(a))),
    ("det", lambda a: g.sum(rad.det(a))),
    ("inv", lambda a: g.sum(g.trace(rad.inv(a) * a * rad.inv(a)))),
]:
    na = rad.node(A, infinitesimal_to_cartesian=False)
    v = fn(na)
    ref = {
        "log_det": g.sum(g.component.log(g.matrix.det(A))),
        "det": g.sum(g.matrix.det(A)),
        "inv": g.sum(g.trace(g.matrix.inv(A))),
    }[name]
    err = abs(complex(v(with_gradients=False)) - complex(ref)) / abs(complex(ref))
    g.message(f"{name}: value {err:.2e}")
    assert err < 1e-12
    Sm = g.component.real(fn(na))
    Sm.functional(na).assert_gradient_error(rng, [A], [A], 1e-5, 1e-8)
    W = g.lattice(A)
    rng.cnormal(W)
    na = rad.node(A, infinitesimal_to_cartesian=False)
    g.component.real(fn(na)).backward(create_graph=True)
    Q = g.component.real(g.inner_product(rad.node(W, with_gradient=False), na.gradient))
    Q.functional(na).assert_gradient_error(rng, [A], [A], 1e-5, 1e-8)

# site-locality: a shifted read of the varying field raises
nodes = [rad.node(v) for v in [x0, C0]]
x = rad.identity(nodes[0])
try:
    rad.jacobian(g.cshift(x, 0, 1) * x, x)
    assert False
except NotImplementedError as e:
    g.message(f"cshift on the path: {e}")
st = g.stencil.matrix(x0, [(0, 0, 0, 0), (1, 0, 0, 0)], [(0, -1, 1.0, [(1, 1, 0), (1, 0, 0)])])
out = x.new()
st(out, x)
try:
    rad.jacobian(out, x)
    assert False
except ValueError as e:
    g.message(f"shifted stencil read on the path: {e}")

# dpt's own local map (the polynomial loop function, the staple through the
# identity boundary): the block and the log-det force against dpt's
even, odd = g.even_odd_projectors(grid)
mu = 1
rho = g.complex(grid)
rho[:] = 0.08
description = [(rho, g.path().f(nu).f(mu).b(nu).b(mu)) for nu in range(4) if nu != mu] + [
    (rho, g.path().b(nu).f(mu).f(nu).b(mu)) for nu in range(4) if nu != mu
]
dpt_params = [rho] + list(poly.parameters())
phi = g.qcd.gauge.smear.directional_parallel_transport(
    U, description, mu, g(even + odd), odd, dpt_params, loop_function=lambda sm, xp: poly([sm], xp[1:])[0]
)
fields = U + dpt_params
nodes = [rad.node(v) for v in fields]
Ux = rad.identity(nodes[mu])
J = rad.jacobian(phi._local_ft(Ux, phi._staple(nodes), nodes[4:]), Ux)
S = g.component.real(g.sum(rad.log_det(J))) * (-1.0)
err = g.norm2(g(odd * J(with_gradients=False)) - phi.jacobian_matrix(fields)) / g.norm2(phi.jacobian_matrix(fields))
g.message(f"dpt: J vs the dpt block: {err:.2e}")
assert err < 1e-24
ga = S.functional(*nodes).gradient(fields, fields)
gb = phi.action_log_det_jacobian().gradient(fields, fields)
err = max(rel(a, b) for a, b in zip(ga, gb))
g.message(f"dpt: log-det force vs dpt: {err:.2e}")
assert err < 1e-26
