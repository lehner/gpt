#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2026
#
# g.local_program: site-local programs (one kernel) against the same
# formulas as plain expressions and as node graphs of ordinary operations:
# values; gradients (derived adjoint program), also with a complex seed and
# restricted by wrt; second and third derivatives (recorded passes: the
# adjoint of the adjoint); outputs of two otypes (one node per otype);
# the tangent program in g.ad.reverse.jacobian (J against the node-graph
# site map, the log-det force against finite differences); a program
# longer than one register field.
#
import gpt as g
import numpy as np

rad = g.ad.reverse
lp = g.local_program
grid = g.grid([4, 4, 4, 8], g.double)
rng = g.random("local_program")

A = g.matrix_color_complex_additive(grid, 3)
x = g.complex(grid)
rng.cnormal([A, x])
one = g.complex(grid)
one[:] = 1.0
M3, S = A.otype, x.otype


def body(A, x, one, trace, conj, real, adj):
    # the same formula symbolically and on nodes (one: 1.0, or a unit field)
    A2 = A * A
    t3 = trace(A2 * A)
    s0 = t3 * conj(x) + trace(A) * (-2.0) + one * 6.0
    s1 = real(x * trace(A2 * adj(A))) + x * x * conj(x)
    M = A2 * x + adj(A) * s0
    return [s0, s1, M]


prog = lp(lambda A, x: body(A, x, 1.0, lp.trace, lp.conj, lp.real, lp.adj), inputs=[M3, S], outputs=[S, S, M3])


def node_body(A, x):
    return body(A, x, rad.node(one, with_gradient=False), g.trace, rad.transform.conj, g.component.real, g.adj)


def rel(a, b):
    return g.norm2(a - b) / g.norm2(b)


# values: plain program against plain expressions
s0, s1, M = prog(A, x)
r0, r1, rM = [v(with_gradients=False) for v in node_body(rad.node(A), rad.node(x))]
err = max(rel(s0, r0), rel(s1, r1), rel(M, rM))
g.message(f"values: {err:.2e}  ({prog.instructions(grid)} instructions)")
assert err < 1e-28

# a real functional of the three outputs
w0, w1, W = g.complex(grid), g.complex(grid), g.lattice(A)
rng.cnormal([w0, w1, W])


def functional(outs):
    s0, s1, M = outs
    c = [rad.node(v, with_gradient=False) for v in [w0, w1, W]]
    return g.component.real(g.sum(g.adj(c[0]) * s0 + g.adj(c[1]) * s1 * s0 + g.trace(g.adj(c[2]) * M)))


def complex_root(outs):
    s0, s1, M = outs
    return g.sum(s0 * s1 + g.trace(M * M))


def leaves():
    return rad.node(A, infinitesimal_to_cartesian=False), rad.node(x)


for label, root in [("real functional", functional), ("complex root, seed 1j", complex_root)]:
    seed = None if root is functional else 1.0j
    nA, nx = leaves()
    root(prog(nA, nx)).backward(initial_gradient=seed)
    ga = [nA.gradient, nx.gradient]
    nA, nx = leaves()
    root(node_body(nA, nx)).backward(initial_gradient=seed)
    gb = [nA.gradient, nx.gradient]
    err = max(rel(a, b) for a, b in zip(ga, gb))
    g.message(f"{label}: gradient vs node graph: {err:.2e}")
    assert err < 1e-26

# restricted by wrt
for which in [0, 1]:
    nA, nx = leaves()
    (ga,) = functional(prog(nA, nx)).backward(wrt=[[nA, nx][which]])
    nA, nx = leaves()
    (gb,) = functional(node_body(nA, nx)).backward(wrt=[[nA, nx][which]])
    err = rel(ga, gb)
    g.message(f"wrt input {which}: {err:.2e}")
    assert err < 1e-26

# against finite differences
nA, nx = leaves()
functional(prog(nA, nx)).functional(nA, nx).assert_gradient_error(rng, [A, x], [A, x], 1e-4, 1e-8)

# second and third derivatives (recorded passes) against the node graph
dA, dx, eA, ex = g.lattice(A), g.lattice(x), g.lattice(A), g.lattice(x)
rng.cnormal([dA, dx, eA, ex])


def contract(nA, nx, a, b):
    return g.component.real(
        g.inner_product(rad.node(a, with_gradient=False), nA.gradient)
        + g.inner_product(rad.node(b, with_gradient=False), nx.gradient)
    )


results = []
for f in [lambda nA, nx: prog(nA, nx), node_body]:
    nA, nx = leaves()
    functional(f(nA, nx)).backward(create_graph=True)
    contract(nA, nx, dA, dx).backward()
    second = [g.copy(nA.gradient), g.copy(nx.gradient)]
    nA, nx = leaves()
    functional(f(nA, nx)).backward(create_graph=True)
    contract(nA, nx, dA, dx).backward(create_graph=True)
    contract(nA, nx, eA, ex).backward()
    results.append(second + [nA.gradient, nx.gradient])
err = max(rel(a, b) for a, b in zip(*results))
g.message(f"second and third derivatives vs node graph: {err:.2e}")
assert err < 1e-24

# a program longer than one register field (an 80-term chain)
def chain(x):
    terms = [x]
    for k in range(80):
        terms.append(terms[-1] * x + 0.01 * k)
    return sum(terms[1:], lp.const(0))


long = lp(chain, inputs=[S], outputs=[S])
y = g.copy(x)
y *= 0.5
(v,) = long(y)
yv = y[:][:, 0]
ts = [yv]
for k in range(80):
    ts.append(ts[-1] * yv + 0.01 * k)
ref = sum(ts[1:])
err = np.max(np.abs(v[:][:, 0] - ref)) / np.max(np.abs(ref))
g.message(f"long program ({long.instructions(grid)} instructions, {long._kernel(grid)[1]} register fields): {err:.2e}")
assert err < 1e-14
ny = rad.node(y)
g.component.real(g.sum(long(ny)[0])).functional(ny).assert_gradient_error(rng, [y], [y], 1e-5, 1e-8)

# the tangent program in g.ad.reverse.jacobian: site maps x -> exp(TA(f(C
# x^dag))) x with f a program (a coefficient field c as second input)
# against the same f as node operations
U = g.qcd.gauge.random(grid, rng, scale=0.5)
x0 = U[0]
C0 = g(0.1 * (U[1] * U[2] + U[3]))
c0 = g.complex(grid)
c0[:] = 0.3 + 0.1j


fprog = lp(
    lambda p, c: p + p * p.adj() * p * c + p * (lp.trace(p * p) * lp.conj(c) * 0.05) + p * 0.1,
    inputs=[M3, S],
    outputs=[M3],
)


def f_program(p, c):
    return fprog(p, c)[0]


def f_nodes(p, c):
    u = rad.node(one, with_gradient=False)
    return p + p * g.adj(p) * p * c + p * (g.trace(p * p) * rad.transform.conj(c) * 0.05) + p * (u * 0.1)


def log_det_action(nodes, fn):
    xi = rad.identity(nodes[0])
    y = g.matrix.exp(g.qcd.gauge.project.traceless_anti_hermitian(fn(nodes[1] * g.adj(xi), nodes[2]))) * xi
    return rad.jacobian(y, xi)


fields = [x0, C0, c0]
J = [log_det_action([rad.node(v) for v in fields], fn)(with_gradients=False) for fn in [f_program, f_nodes]]
err = rel(J[0], J[1])
g.message(f"jacobian with a program site map vs node graph: {err:.2e}")
assert err < 1e-26
nodes = [rad.node(v) for v in fields]
Sld = g.component.real(g.sum(rad.log_det(log_det_action(nodes, f_program)))) * (-1.0)
Sld.functional(*nodes).assert_gradient_error(rng, fields, fields, 1e-4, 1e-8)
