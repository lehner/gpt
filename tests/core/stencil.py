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
grid = g.grid(L, g.double)

# qcd gauge
U = g.qcd.gauge.random(grid, rng)
Udag = [g(g.adj(u)) for u in U]
P = g.copy(U[0])
Ps = g.copy(U[0])


class padded_local_fields:
    def __init__(self, fields, margin):
        fields = g.util.to_list(fields)
        self.grid = fields[0].grid
        self.otype = fields[0].otype
        self.margin = margin
        assert all([f.otype == self.otype for f in fields])
        assert all([f.grid == self.grid for f in fields])

        self.domain = g.domain.local(self.grid, margin)

    def __call__(self, fields):
        return_list = isinstance(fields, list)
        fields = g.util.to_list(fields)
        padded_fields = [self.domain.lattice(self.otype) for f in fields]
        # for now do separate halo exchange for each field, TODO: merge this within domain.project
        for d, s in zip(padded_fields, fields):
            self.domain.project(d, s)
        return padded_fields if return_list else padded_fields[0]

    def extract(self, fields, padded_fields):
        fields = g.util.to_list(fields)
        padded_fields = g.util.to_list(padded_fields)
        for d, s in zip(fields, padded_fields):
            self.domain.promote(d, s)


# test simple cshifts
def stencil_cshift(src, direction):
    p = padded_local_fields(src, [1, 2, 1, 1])

    padded_src = p(src)

    stencil = g.stencil.matrix(
        padded_src,
        [direction],
        [{"target": 0, "accumulate": -1, "adj": 0, "weight": 1.0, "factor": [(1, 0)]}],
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

_P = 0
_U = [1, 2, 3, 4]
_Udag = [5, 6, 7, 8]
_Sp = [1, 2, 3, 4]

code = []
for mu in range(4):
    for nu in range(mu):
        code.append(
            {
                "target": 0,
                "accumulate": -1 if len(code) == 0 else 0,
                "weight": 1.0,
                "adj": 0,
                "factor": [
                    (_U[mu], _P),
                    (_U[nu], _Sp[mu]),
                    (_Udag[mu], _Sp[nu]),
                    (_Udag[nu], _P),
                ],
            }
        )


f_all = [Ps] + U + Udag
p_all = padded_local_fields(f_all, [1, 1, 1, 1])
p = padded_local_fields(P, [1, 1, 1, 1])
padded_all = p_all(f_all)

stencil_plaquette = g.stencil.matrix(
    padded_all[0],
    [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
    code,
)

stencil_plaquette(*padded_all)

p.extract(Ps, padded_all[0])

pval = 2 * g.sum(g.trace(Ps)).real / P.grid.gsites / 4 / 3 / 3

eps = abs(Pref - pval)
g.message(f"Stencil plaquette: {pval} versus reference {Pref}: {eps}")
assert eps < 1e-14


# run again for benchmark:
t = g.timer("test")
t("halo exchange")
padded_all = p_all(f_all)
t("stencil")
stencil_plaquette(*padded_all)
t("extract")
p.extract(Ps, padded_all[0])
t("sum")
pval = 2 * g.sum(g.trace(Ps)).real / P.grid.gsites / 4 / 3 / 3
t()

g.message(t)


# TODO: make halo exchange much more efficient
# each field can have a different margin, also on top and bottom
# do this next

# before this:
#                       : halo exchange        5.56e-03 s (=  88.88 %); time/s = 5.56e-03/5.56e-03/5.56e-03 (min/max/avg)
#                       : stencil              3.60e-04 s (=   5.76 %); time/s = 3.60e-04/3.60e-04/3.60e-04 (min/max/avg)
#                       : sum                  2.27e-04 s (=   3.63 %); time/s = 2.27e-04/2.27e-04/2.27e-04 (min/max/avg)
#                       : extract              1.09e-04 s (=   1.74 %); time/s = 1.09e-04/1.09e-04/1.09e-04 (min/max/avg)
