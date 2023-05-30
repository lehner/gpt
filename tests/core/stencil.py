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
def stencil_cshift(src, direction):
    p = g.padded_local_fields(src, [1, 2, 1, 1])

    padded_src = p(src)

    stencil = g.stencil.matrix(
        padded_src,
        [direction],
        [{"target": 0, "accumulate": -1, "weight": 1.0, "factor": [(1, 0, 0)]}],
    )

    padded_dst = g.lattice(padded_src)
    stencil(padded_dst, padded_src)

    dst = g.lattice(src)
    p.extract(dst, padded_dst)

    return dst


evec = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
for d in range(4):
    Ps1 = stencil_cshift(P, evec[d])
    Ps2 = g.cshift(P, d, 1)
    eps2 = g.norm2(Ps1 - Ps2)

    g.message(f"Test matrix stencil versus cshift in dimension {d}: {eps2}")
    assert eps2 < 1e-13

# test general cshift
Ps1 = stencil_cshift(P, (0, 2, 1, 1))
Ps2 = g.cshift(g.cshift(g.cshift(P, 3, 1), 2, 1), 1, 2)
eps2 = g.norm2(Ps1 - Ps2)
g.message(f"Test matrix stencil verrsus cshift for displacement = (0,2,1,1): {eps2}")
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

stencil_plaquette = g.stencil.matrix(
    padded_P,
    [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
    code,
)

stencil_plaquette(padded_P, *padded_U)

p.extract(Ps, padded_P)

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
