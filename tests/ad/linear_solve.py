#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2026
#
# g.ad.reverse.linear_solve: gradients w.r.t. the right-hand side and the
# operator's parameters (links of a covariant transverse Laplacian), against
# finite differences
#
import gpt as g
import numpy as np

rad = g.ad.reverse
rng = g.random("linear_solve")
grid = g.grid([4, 4, 4, 8], g.double)

mu = 3
dirs = [nu for nu in range(4) if nu != mu]
U = g.qcd.gauge.random(grid, rng)
V = [U[nu] for nu in dirs]


def laplace(m2):
    # (m^2 - Delta_perp) w, Delta_perp the covariant adjoint Laplacian over the
    # directions nu != mu (in the fundamental representation); hermitian and
    # positive
    def op(*args):
        *v, w = args
        r = (m2 + 2 * len(v)) * w
        for vn, nu in zip(v, dirs):
            r = r - vn * g.cshift(w, nu, 1) * g.adj(vn)
            r = r - g.cshift(g.adj(vn) * w * vn, nu, -1)
        return g(r)

    return op


def forward_difference(m2, c, adjoint):
    # m^2 w + c sum_nu (V w(x + nu) V^dag - w) and its adjoint; not hermitian
    def op(*args):
        *v, w = args
        r = (m2 - c * len(v)) * w
        for vn, nu in zip(v, dirs):
            if adjoint:
                r = r + c * g.cshift(g.adj(vn) * w * vn, nu, -1)
            else:
                r = r + c * vn * g.cshift(w, nu, 1) * g.adj(vn)
        return g(r)

    return op


def algebra_field():
    x = g.lattice(grid, U[0].otype.cartesian())
    rng.normal_element(x)
    return x


cg = g.algorithms.inverter.cg(eps=1e-14, maxiter=1000)
fgcr = g.algorithms.inverter.fgcr(eps=1e-14, maxiter=1000, restartlen=20)
op = laplace(0.5)

# plain solve
s = algebra_field()
w = g.ad.reverse.linear_solve(op, s, cg, parameters=V)
eps = (g.norm2(op(*V, w) - s) / g.norm2(s)) ** 0.5
g.message(f"Plain residual: {eps}")
assert eps < 1e-12
assert w.otype.__name__ == s.otype.__name__

# matrix_operators are not supported
try:
    g.ad.reverse.linear_solve(g.matrix_operator(mat=lambda d, x: g.copy(d, x)), s, cg)
    assert False
except NotImplementedError:
    pass

# gradients w.r.t. links and right-hand side, a quadratic and a linear
# functional of the solution (the linear one does not read the value of the
# solve node: its backward solves the forward again)
a = algebra_field()
for name, functional in [
    ("quadratic", lambda w: g.inner_product(a, w).real + 0.3 * g.norm2(w)),
    ("linear", lambda w: g.inner_product(a, w).real),
]:
    aV = [rad.node(x) for x in V]
    a_s = rad.node(s)
    f = functional(g.ad.reverse.linear_solve(op, a_s, cg, parameters=aV)).functional(*aV, a_s)
    for tag, dfields in [("all", V + [s]), ("links", V), ("rhs", [s])]:
        g.message(f"{name} functional, gradient w.r.t. {tag}")
        f.assert_gradient_error(rng, V + [s], dfields, 1e-3, 1e-8)

# a complex functional, real and imaginary parts (conjugate-linear
# convention), with a complex right-hand side (a general color matrix as an
# additive field)
sc = g.mcolor(grid)
rng.cnormal(sc)
ac = g.mcolor(grid)
rng.cnormal(ac)
for seed in [1.0, 1.0j]:
    aV = [rad.node(x) for x in V]
    a_sc = rad.node(g.copy(sc), infinitesimal_to_cartesian=False)
    aw = g.ad.reverse.linear_solve(op, a_sc, cg, parameters=aV)
    c = g.inner_product(ac, aw) + 0.3 * g.inner_product(aw, aw)
    g.message(f"Complex functional, seed {seed}, gradient w.r.t. links")
    (np.conj(seed) * c).real.functional(*aV).assert_gradient_error(rng, V, V, 1e-3, 1e-8)
    c(initial_gradient=seed)
    part = (lambda x: x.real) if seed == 1.0 else (lambda x: x.imag)
    for direction in [1.0, 1.0j]:
        dx = g.mcolor(grid)
        rng.cnormal(dx)
        epsilon = 1e-5
        values = []
        for sign in [1, -1]:
            a_sc.value = g(sc + sign * epsilon * direction * dx)
            values.append(part(c(with_gradients=False)))
        a_sc.value = sc
        num = (values[0] - values[1]) / (2 * epsilon)
        ad = g.inner_product(direction * dx, a_sc.gradient).real
        eps = abs(num - ad) / abs(num)
        g.message(f"Complex functional, seed {seed}, rhs direction {direction}: {eps} ({num}, {ad})")
        assert eps < 1e-8

# a non-hermitian operator: its adjoint is needed for the backward solve
nh = forward_difference(3.0, 0.3, False)
nh_dag = forward_difference(3.0, 0.3, True)
w = g.ad.reverse.linear_solve(nh, s, fgcr, parameters=V)
eps = (g.norm2(nh(*V, w) - s) / g.norm2(s)) ** 0.5
g.message(f"Non-hermitian plain residual: {eps}")
assert eps < 1e-12
x, y = algebra_field(), algebra_field()
eps = abs(g.inner_product(x, nh(*V, y)) - g.inner_product(nh_dag(*V, x), y)) / abs(g.inner_product(x, nh(*V, y)))
g.message(f"Non-hermitian adjoint test: {eps}")
assert eps < 1e-13
for adj_op, expect_correct in [(nh_dag, True), (None, False)]:
    aV = [rad.node(x) for x in V]
    a_s = rad.node(s)
    aw = g.ad.reverse.linear_solve(nh, a_s, fgcr, parameters=aV, adj_op=adj_op)
    f = (g.inner_product(a, aw).real + 0.3 * g.norm2(aw)).functional(*aV, a_s)
    if expect_correct:
        g.message("Non-hermitian operator with its adjoint")
        f.assert_gradient_error(rng, V + [s], V + [s], 1e-3, 1e-8)
    else:
        # treated as hermitian: the gradient must be wrong (the test detects
        # a missing adjoint)
        try:
            g.message("Non-hermitian operator treated as hermitian (expected to fail)")
            f.assert_gradient_error(rng, V + [s], V + [s], 1e-3, 1e-8)
            assert False, "a missing adjoint was not detected"
        except AssertionError as e:
            assert "missing adjoint" not in str(e)
