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


# test general stencil (points off the axes, halo exchange) versus the padded
# stencil (same products, so bit-identical) and cshift
def stencil_reference(fields, points, code):
    fields = [g.copy(f) for f in fields]
    for c in code:
        t = None
        for f, p, a in c["factor"]:
            x = fields[f]
            for d, s in enumerate(points[p]):
                if s != 0:
                    x = g.cshift(x, d, s)
            x = g(g.adj(x)) if a else x
            t = x if t is None else g(t * x)
        t = g(c["weight"] * t)
        if c["accumulate"] != -1:
            t = g(t + fields[c["accumulate"]])
        fields[c["target"]] = t
    return fields


rng_general_stencil = g.random("general stencil")


def test_general_stencil(tag, prec, points, code, n_fields, write_fields, temporaries=(), cse=False):
    grid_prec = g.grid([4, 4, 4, 8], prec)
    fields = [g.mcolor(grid_prec) for i in range(n_fields)]
    rng_general_stencil.element(fields)
    passed = [i for i in range(n_fields) if i not in temporaries]
    ref = stencil_reference(fields, points, code)
    results = []
    matrix_module = sys.modules["gpt.core.stencil.matrix"]
    for padded in [False, True]:
        matrix_module.use_padded = padded
        stencil = g.stencil.matrix(fields[0], points, code, temporaries=temporaries, cse=cse)
        stencil.data_access_hints(write_fields, [i for i in range(len(passed)) if i not in write_fields], [])
        f = [g.copy(fields[i]) for i in passed]
        stencil(*f)
        results.append(f)
    matrix_module.use_padded = False
    for k, i in enumerate(passed):
        eps = (g.norm2(results[0][k] - ref[i]) / g.norm2(ref[i])) ** 0.5
        diff = g.norm2(results[0][k] - results[1][k])
        g.message(
            f"Test general stencil ({tag}, {prec.__name__}) field {i}: versus cshift {eps}, versus padded {diff}"
        )
        assert eps < prec.eps * 100
        assert diff == 0.0


for prec in [g.double, g.single]:
    test_general_stencil(
        "diagonal points",
        prec,
        [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0), (-1, 0, 0, 1), (0, 0, -2, 3), (2, -1, 1, -1)],
        [
            {"target": 0, "accumulate": -1, "weight": 1.0, "factor": [(1, 0, 0), (2, 1, 0), (1, 2, 1), (2, 0, 1)]},
            {"target": 0, "accumulate": 0, "weight": 0.5j, "factor": [(1, 3, 0), (2, 4, 1)]},
            {"target": 3, "accumulate": -1, "weight": 2.0, "factor": [(2, 5, 0), (1, 6, 1), (2, 3, 0)]},
        ],
        4,
        [0, 3],
    )
    test_general_stencil(
        "long shifts",
        prec,
        [(0, 0, 0, 0), (5, 0, 0, 0), (-3, 4, 0, 0), (0, 0, 7, -9), (4, 4, 4, 8)],
        [{"target": 0, "accumulate": -1, "weight": 1.0, "factor": [(1, 1, 0), (1, 2, 1), (1, 3, 0), (1, 4, 1)]}],
        2,
        [0],
    )
    # points that skip ranks and wind around the lattice several times
    test_general_stencil(
        "windings",
        prec,
        [(0, 0, 0, 0), (0, 0, 0, 9), (0, 0, 0, -13), (1, -2, 3, -17), (0, 0, 0, 16), (13, -9, 6, -21)],
        [{"target": 0, "accumulate": -1, "weight": 1.0, "factor": [(1, 1, 0), (1, 2, 1), (1, 3, 0), (1, 4, 1), (1, 5, 0)]}],
        2,
        [0],
    )
    # more fields read at shifted points than one communication batch holds
    n_many = 40
    test_general_stencil(
        "many fields",
        prec,
        [(0, 0, 0, 0)] + [(1, (i % 3) - 1, 0, (i % 5) - 2) for i in range(n_many)],
        [
            {"target": 0, "accumulate": -1 if i == 0 else 0, "weight": 1.0, "factor": [(1 + i, 1 + i, 0)]}
            for i in range(n_many)
        ],
        1 + n_many,
        [0],
    )
    for cse in [False, True]:
        test_general_stencil(
            f"temporaries, cse = {cse}",
            prec,
            [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (-1, 1, 0, 0), (-1, 0, 0, 0)],
            [
                {"target": 3, "accumulate": -1, "weight": 1.0, "factor": [(1, 1, 0), (2, 2, 1), (1, 0, 1)]},
                {"target": 0, "accumulate": -1, "weight": 1.0, "factor": [(3, 0, 0), (2, 0, 0)]},
                {"target": 0, "accumulate": 0, "weight": 1.0, "factor": [(3, 0, 1), (1, 3, 0), (2, 4, 1)]},
            ],
            4,
            [0],
            temporaries=[3],
            cse=cse,
        )


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
    g.message("Before stencil.tensor")
    ein(R, Q1, Q2)
    g.message("After stencil.tensor")
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

# test combination of checkerpointing and simd
L = [16, 12, 24, 32]
grid = g.grid(L, g.single, g.redblack)
grid = grid.split([1]*4, L) # create a local grid
test = g.complex(grid)
rng.cnormal(test)
for mu in range(4):
    for nu in range(4):

        if mu == nu:
            continue

        dd = [0] * 4
        dd[mu] = -2
        dd[nu] = -2
        dd = tuple(dd)

        for p in [g.even, g.odd]:
            test.checkerboard(p)

            st = g.local_stencil.matrix_vector(
                test,
                test,
                [(0, 0, 0, 0), dd],
                [
                    {
                        "target": 0,
                        "source": 1,
                        "source_point": 1,
                        "accumulate": -1,
                        "weight": -2.0,
                        "factor": [],
                    }
                ],
                vector_parity=p.tag,
            )

            reference = g(-2.0 * g.cshift(g.cshift(test, mu, -2), nu, -2))
            out = g.lattice(test)  # 0: 00, 1: 20, 2: 11, 3: 31, 4: 02, 5: 22, 6: 13, 7: 33
            st([test], [out, test])

            eps2 = g.norm2(reference - out) / g.norm2(reference)
            g.message(f"{p.__name__} {mu} {nu} {eps2}")
            assert eps2 < 1e-12


# local matrix stencils with per-site temporaries: the same code with its
# temporaries owned by the stencil (cache-blocked, not passed by the caller)
# must give bitwise identical results, for several block sizes including one
# that does not divide the volume
for precision in [g.double, g.single]:
    grid = g.grid([4, 4, 4, 8], precision)
    U = g.qcd.gauge.random(grid, rng)
    A, B = rng.cnormal([g.mcolor(grid), g.mcolor(grid)])
    points = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, -1)]
    # fields: 0 = output, 1-3 = temporaries, 4 = A, 5 = B, 6 = U_0, 7 = U_3
    code = [
        (1, -1, 0.5, [(4, 0, 0), (5, 0, 0)]),  # t1 = 0.5 A B
        (2, -1, 1.0, [(1, 0, 1), (6, 1, 0)]),  # t2 = t1^dag U_0(x + 0)
        (2, 2, -0.25j, [(7, 2, 0), (4, 0, 1)]),  # t2 += c U_3(x - 3) A^dag
        (3, -1, 1.0, [(2, 0, 0), (1, 0, 0)]),  # t3 = t2 t1
        (3, 3, 2.0, [(5, 0, 1)]),  # t3 += 2 B^dag
        (0, -1, 1.0, [(3, 0, 0), (2, 0, 1)]),  # out = t3 t2^dag
        (0, 0, 0.5, [(1, 0, 0)]),  # out += 0.5 t1
    ]
    def fresh_fields(temporaries):
        # new output (and temporaries, if passed) for every run, shared
        # read-only inputs
        return [g.lattice(A) for _ in range(1 + temporaries)] + [A, B, U[0], U[3]]

    f_ref = fresh_fields(3)
    g.local_stencil.matrix(A, points, code)(*f_ref)
    for block in [0, 1, 7, 64, 10**6]:
        f = fresh_fields(0)
        g.local_stencil.matrix(A, points, code, temporaries=[1, 2, 3], osites_per_cache_block=block)(*f)
        ok = np.array_equal(f[0][:], f_ref[0][:])
        g.message(f"local stencil with temporaries, {precision.__name__}, block {block}: {'ok' if ok else 'MISMATCH'}")
        assert ok

# common-subexpression elimination of the executed plan (cse=): the same
# results up to rounding.  The code repeats pairs (also as their adjoint
# reversal, and inside a repeated triple, so temporaries are built from
# temporaries) and reads fields the kernel writes -- a passed field W and a
# temporary t, each rewritten between reads of the same pair -- which may
# only be combined per write version.
for precision in [g.double, g.single]:
    grid = g.grid([4, 4, 4, 8], precision)
    U = g.qcd.gauge.random(grid, rng)
    A, B, W0 = rng.cnormal([g.mcolor(grid), g.mcolor(grid), g.mcolor(grid)])
    points = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)]
    # fields: 0 = out, 1 = W (read and rewritten), 2 = A, 3 = B, 4 = U_0,
    # 5 = U_1, 6 = temporary t
    code = [
        (6, -1, 1.0, [(2, 0, 0), (3, 0, 0)]),  # t = A B
        (0, -1, 1.0, [(2, 0, 0), (4, 1, 0), (5, 2, 1), (1, 0, 0), (6, 0, 0)]),
        (0, 0, 0.5j, [(5, 2, 0), (4, 1, 1), (2, 0, 1), (1, 0, 0), (6, 0, 0)]),
        (1, 1, 1.0, [(1, 0, 0), (3, 0, 0)]),  # W += W B
        (6, 6, -1.0, [(6, 0, 0), (2, 0, 0)]),  # t -= t A
        (0, 0, 2.0, [(2, 0, 0), (4, 1, 0), (5, 2, 1), (1, 0, 0), (6, 0, 0)]),
        (0, 0, 1.0, [(1, 0, 0), (6, 0, 0), (2, 0, 0), (4, 1, 0)]),
        (1, 1, 0.25, [(1, 0, 0), (6, 0, 0), (2, 0, 0), (4, 1, 0)]),
    ]
    res = []
    for cse in [False, True]:
        W = g.copy(W0)
        out = g.lattice(A)
        K = g.local_stencil.matrix(A, points, code, temporaries=[6], cse=cse)
        if cse:
            assert K.executed is not None and len(K.executed[2]) > 2
        K(out, W, A, B, U[0], U[1])
        res.append((out, W))
    eps = max(
        (g.norm2(x - y) / g.norm2(x)) ** 0.5 for x, y in zip(res[0], res[1])
    )
    g.message(f"local stencil cse, {precision.__name__}: {eps}")
    assert eps < precision.eps * 100
