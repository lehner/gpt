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
grid = g.grid([4, 4, 4, 8], g.double)
rng = g.random("jacobian")
U = g.qcd.gauge.random(grid, rng, scale=0.5)
x0 = U[0]
# (a staple-like constant, scaled as by rho: the site maps stay invertible)
C0 = g(0.1 * (U[1] * U[2] + U[3]))
T = g.group.algebra_kernels(grid, x0.otype.cartesian()).field_generators
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
        y.backward(
            initial_gradient=g.cartesian_to_infinitesimal(yv, Tb),
            retain_values=True,
            wrt=[nodes[0]],
        )
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
    g.message(
        f"{name}: chunk 3 vs unchunked force: {err:.2e}, values {abs(fS(fields) - fSc(fields)):.2e}"
    )
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

# dpt (on g.ad.reverse.jacobian since it uses it) against its generic path
# (8 recorded reverse passes through the whole transport): the polynomial
# loop function with fixed loops (the 2x1 rectangles around U_0(x)), the
# block, the log-det force, the weighted force with a prescribed staple, the
# chunked replays, and the preimage (exact block-triangular solve) against
# the fgcr one
even, odd = g.even_odd_projectors(grid)
mu = 0
rho = g.complex(grid)
rho[:] = 0.08
description = [(rho, g.path().f(nu).f(mu).b(nu).b(mu)) for nu in range(4) if nu != mu] + [
    (rho, g.path().b(nu).f(mu).f(nu).b(mu)) for nu in range(4) if nu != mu
]
loops = [g.path().f(nu).f(0).b(nu).b(nu).b(0).f(nu) for nu in range(1, 4)]
dpt_params = [rho] + list(poly.parameters())


def loop_function(sm, xp, L):
    f = poly([sm], xp[1:])[0]
    for l in L:
        f = f + sm * l * 0.05
    return f


def dpt(P1, chunk=4):
    return g.qcd.gauge.smear.directional_parallel_transport(
        U,
        description,
        mu,
        g(even + odd),
        P1,
        dpt_params,
        loop_function=loop_function,
        loops=loops,
        jacobian_chunk=chunk,
    )


fields = U + dpt_params
for P1, name in [(even, "even"), (odd, "odd")]:
    phi = dpt(P1)
    M, Mg = phi.jacobian_matrix(fields), phi._jacobian_matrix_generic(fields)[0]
    err = g.norm2(M - Mg) / g.norm2(Mg)
    g.message(f"dpt {name}: block vs generic: {err:.2e}")
    assert err < 1e-26
    ga = phi.action_log_det_jacobian().gradient(fields, fields)
    gb = phi._action_log_det_jacobian_gradient_generic(fields, fields)
    err = max(rel(a, b) for a, b in zip(ga, gb))
    g.message(f"dpt {name}: log-det force vs generic: {err:.2e}")
    assert err < 1e-24
    gc = dpt(P1, None).action_log_det_jacobian().gradient(fields, fields)
    err = max(rel(a, b) for a, b in zip(ga, gc))
    g.message(f"dpt {name}: chunk 4 vs unchunked: {err:.2e}")
    assert err < 1e-26
    # the weighted force with a prescribed staple (as in a barrier: a
    # function of the weights only), against differences
    W = [g.qcd.gauge.random(grid, rng)[0] for _ in range(6)]
    staple = lambda fs: sum(fs[4] * w for w in W)
    weight = g.complex(grid)
    rng.uniform_real(weight)

    class weighted(g.group.differentiable_functional):
        def __call__(self, fs):
            ld = phi.log_det_jacobian_field(fs, staple)
            return -g.sum(g(weight * ld)).real

        def gradient(self, fs, dfs):
            return phi.weighted_log_det_jacobian_gradient(fs, dfs, weight, staple)

    weighted().assert_gradient_error(rng, fields, fields[4:], 1e-5, 1e-8)
    # the preimage of a node: exact solve against fgcr
    y = phi(fields)
    nodes = [rad.node(v) for v in y]
    W2 = [g.lattice(u) for u in U]
    for w in W2:
        rng.cnormal(w)

    def cost(inverse):
        nodes = [rad.node(v) for v in y]
        x = inverse(nodes)
        L = sum(g.inner_product(rad.node(w, with_gradient=False), xn) for w, xn in zip(W2, x[0:4]))
        return g.component.real(L).functional(*nodes)

    def preimage_fgcr(n):
        (u,) = rad.preimage(phi, n, [mu], lambda v: phi.inv(v, max_iter=1000))
        return [u if i == mu else x for i, x in enumerate(n)]

    exact = cost(lambda n: phi.inv(n, max_iter=1000))
    fgcr = cost(preimage_fgcr)
    ga = exact.gradient(y, y)
    gb = fgcr.gradient(y, y)
    err = max(rel(a, b) for a, b in zip(ga, gb))
    g.message(f"dpt {name}: preimage, exact vs fgcr: {err:.2e}")
    assert err < 1e-20
    exact.assert_gradient_error(rng, y, y, 1e-5, 1e-7)
