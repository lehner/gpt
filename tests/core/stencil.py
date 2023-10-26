#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2023
#
import gpt as g
import numpy as np
import sys

# random
rng = g.random("test")

# grid
L = [8, 12, 24, 24]
# L = [32,32,32,32]
grid = g.grid(L, g.double)

# qcd gauge
U = g.qcd.gauge.random(grid, rng)
Udag = [g(g.adj(u)) for u in U]
P = g.copy(U[0])
Ps = g.copy(U[0])


# test simple cshifts
def stencil_cshift(src, direction1, direction2):
    stencil = g.stencil.matrix(
        src,
        [direction1, direction2, (0, 0, 0, 0)],
        [
            {"target": 0, "accumulate": -1, "weight": 1.0, "factor": [(1, 0, 0)]},
            {"target": 0, "accumulate": 0, "weight": 1.0, "factor": [(2, 1, 0)]},
            {"target": 0, "accumulate": 0, "weight": 1.0, "factor": [(2, 2, 0)]},
        ],
    )
    stencil.data_access_hints([0], [1, 2, 3, 4], [])
    dst = g.lattice(src)
    stencil(dst, src, src)
    return dst


evec = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
for d1 in range(4):
    for d2 in range(d1):
        Ps1 = stencil_cshift(P, evec[d1], evec[d2])
        Ps2 = g.cshift(P, d1, 1)
        Ps2 += g.cshift(P, d2, 1)
        Ps2 += P
        eps2 = g.norm2(Ps1 - Ps2)

        g.message(f"Test matrix stencil versus cshift in dimension {d1} x {d2}: {eps2}")
        assert eps2 < 1e-13

# test general cshift
Ps1 = stencil_cshift(P, (0, 2, 1, 1), (0, 0, 0, 0))
Ps2 = g(g.cshift(g.cshift(g.cshift(P, 3, 1), 2, 1), 1, 2) + 2.0 * P)
eps2 = g.norm2(Ps1 - Ps2)
g.message(f"Test matrix stencil versus cshift for displacement = (0,2,1,1): {eps2}")
assert eps2 < 1e-25


# test stencil implementation of plaquette
Pref = 0.7980707694878268
# g.qcd.gauge.plaquette(U)

_P = 0
_U = [1, 2, 3, 4]
_Sp = [1, 2, 3, 4]

code = []
for mu in range(4):
    for nu in range(mu):
        code.append(
            {
                "target": 0,
                "accumulate": -1 if len(code) == 0 else 0,
                "weight": 1.0,
                "factor": [
                    (_U[mu], _P, 0),
                    (_U[nu], _Sp[mu], 0),
                    (_U[mu], _Sp[nu], 1),
                    (_U[nu], _P, 1),
                ],
            }
        )


p_U = g.padded_local_fields(U, [1, 1, 1, 1])
p = g.padded_local_fields(P, [1, 1, 1, 1])

padded_U = p_U(U)
padded_P = p(P)

stencil_plaquette = g.local_stencil.matrix(
    padded_P,
    [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
    code,
)

stencil_plaquette(padded_P, *padded_U)

p.extract(Ps, padded_P)

pval = 2 * g.sum(g.trace(Ps)).real / P.grid.gsites / 4 / 3 / 3

eps = abs(Pref - pval)
g.message(f"Stencil plaquette (local + padding): {pval} versus reference {Pref}: {eps}")
assert eps < 1e-14

stencil_plaquette = g.stencil.matrix(
    P,
    [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
    code,
)

stencil_plaquette(Ps, *U)
pval = 2 * g.sum(g.trace(Ps)).real / P.grid.gsites / 4 / 3 / 3

eps = abs(Pref - pval)
g.message(f"Stencil plaquette: {pval} versus reference {Pref}: {eps}")
assert eps < 1e-14


# run again for benchmark:
# t = g.timer("test")
# t("halo exchange")
# padded_U = p_U(U)
# t("stencil")
# stencil_plaquette(padded_P, *padded_U)
# t("extract")
# p.extract(Ps, padded_P)
# t("sum")
# pval = 2 * g.sum(g.trace(Ps)).real / P.grid.gsites / 4 / 3 / 3
# t("reference")
# pvalref = g.qcd.gauge.plaquette(U)
# t()

# g.message(t)


# before this:
#                       : halo exchange        5.56e-03 s (=  88.88 %); time/s = 5.56e-03/5.56e-03/5.56e-03 (min/max/avg)
#                       : stencil              3.60e-04 s (=   5.76 %); time/s = 3.60e-04/3.60e-04/3.60e-04 (min/max/avg)
#                       : sum                  2.27e-04 s (=   3.63 %); time/s = 2.27e-04/2.27e-04/2.27e-04 (min/max/avg)
#                       : extract              1.09e-04 s (=   1.74 %); time/s = 1.09e-04/1.09e-04/1.09e-04 (min/max/avg)

# after merging copy plans, only marginal improvement: must have other bottleneck
#                       : halo exchange        4.85e-03 s (=  88.69 %); time/s = 4.85e-03/4.85e-03/4.85e-03 (min/max/avg)
#                       : stencil              3.05e-04 s (=   5.57 %); time/s = 3.05e-04/3.05e-04/3.05e-04 (min/max/avg)
#                       : sum                  2.09e-04 s (=   3.83 %); time/s = 2.09e-04/2.09e-04/2.09e-04 (min/max/avg)
#                       : extract              1.04e-04 s (=   1.91 %); time/s = 1.04e-04/1.04e-04/1.04e-04 (min/max/avg)

# next: only halo exchange for minimal fields; some improvement but still not acceptable
#                       : halo exchange        3.21e-03 s (=  85.31 %); time/s = 3.21e-03/3.21e-03/3.21e-03 (min/max/avg)
#                       : stencil              2.76e-04 s (=   7.34 %); time/s = 2.76e-04/2.76e-04/2.76e-04 (min/max/avg)
#                       : sum                  2.10e-04 s (=   5.57 %); time/s = 2.10e-04/2.10e-04/2.10e-04 (min/max/avg)
#                       : extract              6.75e-05 s (=   1.79 %); time/s = 6.75e-05/6.75e-05/6.75e-05 (min/max/avg)

# for 32^4 global lattice it looks better:
#                       : halo exchange        1.10e-02 s (=  68.92 %); time/s = 1.10e-02/1.10e-02/1.10e-02 (min/max/avg)
#                       : stencil              3.46e-03 s (=  21.79 %); time/s = 3.46e-03/3.46e-03/3.46e-03 (min/max/avg)
#                       : extract              8.28e-04 s (=   5.20 %); time/s = 8.28e-04/8.28e-04/8.28e-04 (min/max/avg)
#                       : sum                  6.51e-04 s (=   4.09 %); time/s = 6.51e-04/6.51e-04/6.51e-04 (min/max/avg)

# and for same volume comparison with reference cshift implementation:
#                       : reference            6.63e-01 s (=  97.86 %); time/s = 6.63e-01/6.63e-01/6.63e-01 (min/max/avg)
#                       : halo exchange        9.29e-03 s (=   1.37 %); time/s = 9.29e-03/9.29e-03/9.29e-03 (min/max/avg)
#                       : stencil              3.50e-03 s (=   0.52 %); time/s = 3.50e-03/3.50e-03/3.50e-03 (min/max/avg)
#                       : sum                  8.67e-04 s (=   0.13 %); time/s = 8.67e-04/8.67e-04/8.67e-04 (min/max/avg)
#                       : extract              8.11e-04 s (=   0.12 %); time/s = 8.11e-04/8.11e-04/8.11e-04 (min/max/avg)


# now test matrix_vector
v = g.vspincolor(grid)
m = g.mcolor(grid)
nevec = [tuple([-x for x in y]) for y in evec]
src = g.vspincolor(grid)
rng.cnormal(src)
cov = g.covariant.shift(U, boundary_phases=[1.0, 1.0, 1.0, 1.0])
for mu in range(4):
    st = g.stencil.matrix_vector(
        U[0],
        src,
        [(0, 0, 0, 0), evec[mu], nevec[mu]],
        [
            {
                "target": 0,
                "source": 1,
                "source_point": 0,
                "accumulate": -1,
                "weight": -2.0,
                "factor": [],
            },
            {
                "target": 0,
                "source": 1,
                "source_point": 1,
                "accumulate": 0,
                "weight": 1.0,
                "factor": [(mu, 0, 0)],
            },
            {
                "target": 0,
                "source": 1,
                "source_point": 2,
                "accumulate": 0,
                "weight": 1.0,
                "factor": [(mu, 2, 1)],
            },
        ],
    )

    def lap(dst, src):
        dst @= -2.0 * src + cov.forward[mu] * src + cov.backward[mu] * src

    ref = g.lattice(src)
    stv = g.lattice(src)

    lap(ref, src)
    st(U, [stv, src])

    eps2 = g.norm2(stv - ref)
    g.message(f"Stencil covariant laplace versus cshift version: {eps2}")
    assert eps2 < 1e-25


# tensor stencil test for case of diquark
def serial_diquark(Q1, Q2):
    eps = g.epsilon(Q1.otype.shape[2])
    R = g.lattice(Q1)

    # D_{a2,a1} = epsilon_{a1,b1,c1}*epsilon_{a2,b2,c2}*Q1_{b1,b2}*spin_transpose(Q2_{c1,c2})
    Q1 = g.separate_color(Q1)
    Q2 = g.separate_color(Q2)

    D = {x: g.lattice(Q1[x]) for x in Q1}
    for d in D:
        D[d][:] = 0

    for i1, sign1 in eps:
        for i2, sign2 in eps:
            D[i2[0], i1[0]] += sign1 * sign2 * Q1[i1[1], i2[1]] * g.transpose(Q2[i1[2], i2[2]])

    g.merge_color(R, D)
    return R


def stencil_diquark(Q1, Q2):
    Nc = Q1.otype.shape[2]
    Ns = Q1.otype.shape[0]
    eps = g.epsilon(Nc)
    R = g.mspincolor(grid)
    code = []
    acc = {}
    ti = g.stencil.tensor_instructions
    for i in range(Ns):
        for j in range(Ns):
            for l in range(Ns):
                for i1, sign1 in eps:
                    for i2, sign2 in eps:
                        dst = (i * Ns + j) * Nc * Nc + i2[0] * Nc + i1[0]
                        aa = (Ns * i + l) * Nc * Nc + i1[1] * Nc + i2[1]
                        bb = (Ns * j + l) * Nc * Nc + i1[2] * Nc + i2[2]
                        if dst not in acc:
                            acc[dst] = True
                            mode = ti.mov if sign1 * sign2 > 0 else ti.mov_neg
                        else:
                            mode = ti.inc if sign1 * sign2 > 0 else ti.dec
                        code.append((0, dst, mode, 1.0, [(1, 0, aa), (2, 0, bb)]))

    segments = [(len(code) // (Ns * Ns), Ns * Ns)]
    ein = g.stencil.tensor(Q1, [(0, 0, 0, 0)], code, segments)
    ein(R, Q1, Q2)
    return R


Q1 = g.mspincolor(grid)
Q2 = g.mspincolor(grid)

rng.cnormal([Q1, Q2])
st_di = stencil_diquark(Q1, Q2)
se_di = serial_diquark(Q1, Q2)
std_di = g.qcd.baryon.diquark(Q1, Q2)
eps2 = g.norm2(st_di - se_di) / g.norm2(se_di)
g.message(f"Diquark stencil test (stencil <> serial): {eps2}")
assert eps2 < 1e-25

eps2 = g.norm2(st_di - std_di) / g.norm2(std_di)
g.message(f"Diquark stencil test (stencil <> g.qcd.gauge.diquark): {eps2}")
assert eps2 < 1e-25

# and use this to test einsum
# D_{a2,a1} = epsilon_{a1,b1,c1}*epsilon_{a2,b2,c2}*Q1_{b1,b2}*spin_transpose(Q2_{c1,c2})
einsum_di = g.einsum("acd,bef,ACce,BCdf->ABba", g.epsilon, g.epsilon, Q1, Q2, Q1)
es_di = einsum_di(Q1, Q2)

eps2 = g.norm2(st_di - es_di) / g.norm2(st_di)
g.message(f"Diquark stencil test (stencil <> einsum): {eps2}")
assert eps2 < 1e-25

einsum_trace = g.einsum("AAaa->", Q1, g.complex(Q1.grid))
xx = einsum_trace(Q1)
yy = g(g.trace(Q1))
eps2 = g.norm2(xx - yy) / g.norm2(yy)
g.message(f"Einsum trace test: {eps2}")
assert eps2 < 1e-25

einsum_spintrace = g.einsum("AAab->ab", Q1, g.mcolor(Q1.grid))
xx = einsum_spintrace(Q1)
yy = g(g.spin_trace(Q1))
eps2 = g.norm2(xx - yy) / g.norm2(yy)
g.message(f"Einsum spintrace test: {eps2}")
assert eps2 < 1e-25

einsum_transpose = g.einsum("ABab->BAba", Q1, Q1)
xx = einsum_transpose(Q1)
yy = g(g.transpose(Q1))
eps2 = g.norm2(xx - yy) / g.norm2(yy)
g.message(f"Einsum transpose test: {eps2}")
assert eps2 < 1e-25

einsum_mm = g.einsum("ABab,BCbc->ACac", Q1, Q1, Q1)
xx = einsum_mm(Q1, Q2)
yy = g(Q1 * Q2)
eps2 = g.norm2(xx - yy) / g.norm2(yy)
g.message(f"Einsum mm test: {eps2}")
assert eps2 < 1e-25
