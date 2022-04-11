#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import numpy as np

grid = g.grid([8, 4, 4, 4], g.double)
rng = g.random("test")
dims = ["X", "Y", "Z", "T"]

for lat in [g.mspincolor, g.mspin]:
    l = lat(grid)
    g.message(lat.__name__)
    rng.cnormal(l)

    dst = g.lattice(l)
    ref = g.lattice(l)

    assert g.norm2(l) > 1e-7

    for i, d1 in enumerate(dims):
        for j, d2 in enumerate(dims):
            if i < j:
                dst @= g.gamma[d1] * g.gamma[d2] * l
                dst -= g.gamma[d2] * g.gamma[d1] * l
                dst *= 1 / 2.0
                ref @= g.gamma["Sigma%s%s" % (d1, d2)] * l
                eps = g.norm2(dst - ref) / g.norm2(l)
                g.message("Test Sigma%s%s: " % (d1, d2), eps)
                assert eps == 0.0

                dst @= l * g.gamma[d1] * g.gamma[d2]
                dst -= l * g.gamma[d2] * g.gamma[d1]
                dst *= 1 / 2.0
                ref @= l * g.gamma["Sigma%s%s" % (d1, d2)]
                eps = g.norm2(dst - ref) / g.norm2(l)
                g.message("Test Rev Sigma%s%s: " % (d1, d2), eps)
                assert eps == 0.0

    dst @= g.gamma["X"] * g.gamma["Y"] * g.gamma["Z"] * g.gamma["T"] * l
    ref @= g.gamma[5] * l
    eps = g.norm2(dst - ref) / g.norm2(l)
    g.message("Test Gamma5: ", eps)
    assert eps == 0.0

    # test creating spinor projections as expession templates
    P = g.gamma[0] * g.gamma[1] - g.gamma[2] * g.gamma[3]

    dst @= P * l
    ref @= g.gamma[0] * g.gamma[1] * l - g.gamma[2] * g.gamma[3] * l
    eps = g.norm2(dst - ref) / g.norm2(l)
    g.message("Test Regular Expression: ", eps)
    assert eps == 0.0

    # test algebra versus matrix
    for mu in [0, 1, 2, 3, 5, "I"]:
        for op in [
            lambda a, b: a * b,
            lambda a, b: b * a,
            lambda a, b: g.spin_trace(a * b),
            lambda a, b: g.spin_trace(b * a),
            lambda a, b: g.color_trace(a * b),
            lambda a, b: g.color_trace(b * a),
            lambda a, b: g.trace(a * b),
            lambda a, b: g.trace(b * a),
        ]:
            dst_alg = g(op(g.gamma[mu], l))
            dst_mat = g(op(g.gamma[mu].tensor(), l))
            eps2 = g.norm2(dst_alg - dst_mat) / g.norm2(dst_mat)
            g.message(f"Algebra<>Matrix {mu}: {eps2}")
            assert eps2 < 1e-14

# reconstruct and test the gamma matrix elements
for mu in g.gamma:
    gamma = g.gamma[mu]
    g.message("Test numpy matrix representation of", mu)
    gamma_mu_mat = np.identity(4, dtype=np.complex128)
    for j in range(4):
        c = g.vspin([1 if i == j else 0 for i in range(4)])
        gamma_mu_mat[:, j] = (gamma * c).array
    eps = np.linalg.norm(gamma_mu_mat - gamma.tensor().array)
    assert eps < 1e-14

# test multiplication of spin-color vector
sc = g.vspincolor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.12, 0]])
for mu in g.gamma:
    gamma = g.gamma[mu]
    g.message("Test numpy matrix application to vspincolor for", mu)
    vec = (gamma * sc).array
    for i in range(4):
        for j in range(3):
            eps = abs(vec[i, j] - sum(gamma.tensor().array[i, :] * sc[:, j]))
            assert eps < 1e-14

# test multiplication of spin vector
sc = g.vspin([1, 3.3, 4.7, 0.2391])
for mu in g.gamma:
    gamma = g.gamma[mu]
    g.message("Test numpy matrix application to vspin for", mu)
    vec = (gamma * sc).array
    for i in range(4):
        eps = abs(vec[i] - sum(gamma.tensor().array[i, :] * sc[:]))
        assert eps < 1e-14

# test multiplication with spin-color matrix (propagator)
l = g.mspincolor(grid)
rng.cnormal(l)
prop = l[0, 0, 0, 0]
propPrime = g.gamma[5] * prop * g.gamma[5]
g.message("Test multiplication with spin-color matrix")
for s1 in range(4):
    for s2 in range(4):
        for c1 in range(3):
            for c2 in range(3):
                eps = abs(
                    prop[s1, s2, c1, c2] / propPrime[s1, s2, c1, c2] - (-1) ** (s1 // 2 + s2 // 2)
                )
                assert eps < 1e-14


# test adjoint applied to lattice
for gg in g.gamma:
    l2 = g(g.adj(g.gamma[gg]) * l)
    l2p = g(g.adj(g.gamma[gg].tensor()) * l)
    eps = g.norm2(l2 - l2p) ** 0.5
    g.message(f"Adjoint test {gg}: {eps}")
    assert eps < 1e-13

g.message("All tests passed")
