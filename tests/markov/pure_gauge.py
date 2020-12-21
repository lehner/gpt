#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import numpy as np
import sys

# grid
L = [8, 8, 8, 8]
grid = g.grid(L, g.single)
grid_eo = g.grid(L, g.single, g.redblack)

# hot start
g.default.push_verbose("random", False)
rng = g.random("test", "vectorized_ranlux24_24_64")
U = g.qcd.gauge.unit(grid)
Nd = len(U)

# red/black mask
mask_rb = g.complex(grid_eo)
mask_rb[:] = 1

# full mask
mask = g.complex(grid)

# simple plaquette action
def staple(U, mu):
    st = g.lattice(U[0])
    st[:] = 0
    Nd = len(U)
    for nu in range(Nd):
        if mu != nu:
            st += g.qcd.gauge.staple(U, mu, nu) / U[0].otype.Nc
    return st


# 10 metropolis sweeps
g.default.push_verbose("metropolis", False)
markov = g.algorithms.markov.metropolis(rng, step_size=0.5)
beta = 5.5

for it in range(10):
    plaq = g.qcd.gauge.plaquette(U)
    g.message(f"Metropolis {it} has P = {plaq}")
    for cb in [g.even, g.odd]:
        mask[:] = 0
        mask_rb.checkerboard(cb)
        g.set_checkerboard(mask, mask_rb)

        for mu in range(Nd):
            st = g.eval(beta * staple(U, mu))
            markov(U[mu], st, mask)

assert abs(plaq - 0.9118731021881104) < 1e-6

# 5 heatbath sweeps
g.default.push_verbose("su2_heat_bath", False)
markov = g.algorithms.markov.su2_heat_bath(rng)
U = g.qcd.gauge.unit(grid)
for it in range(5):
    plaq = g.qcd.gauge.plaquette(U)
    g.message(f"SU(2)-subgroup heatbath {it} has P = {plaq}")
    for cb in [g.even, g.odd]:
        mask[:] = 0
        mask_rb.checkerboard(cb)
        g.set_checkerboard(mask, mask_rb)

        for mu in range(Nd):
            st = g.eval(beta * staple(U, mu))
            markov(U[mu], st, mask)

assert abs(plaq - 0.5596460567580329) < 1e-6
