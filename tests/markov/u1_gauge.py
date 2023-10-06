#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Tilo Wettig 2020
#
import gpt as g
import numpy as np
import sys

# grid
L = [8, 8, 8, 8]
grid = g.grid(L, g.single)
grid_eo = g.grid(L, g.single, g.redblack)

# cold start
g.default.push_verbose("random", False)
rng = g.random("test", "vectorized_ranlux24_24_64")
U = [g.complex(grid) for i in range(4)]
for mu in range(len(U)):
    U[mu][:] = 1

# red/black mask
mask_rb = g.complex(grid_eo)
mask_rb[:] = 1

# full mask
mask = g.complex(grid)


# simple plaquette action
def staple(U, mu):
    st = g.lattice(U[0])
    st[:] = 0
    for nu in range(len(U)):
        if mu != nu:
            st += g.qcd.gauge.staple(U, mu, nu)
    return st


# 5 heatbath sweeps
beta = 1.1
g.default.push_verbose("u1_heat_bath", False)
markov = g.algorithms.markov.u1_heat_bath(rng)
for it in range(5):
    plaq = g.qcd.gauge.plaquette(U)
    g.message(f"U(1) heatbath {it} has P = {plaq}")
    for cb in [g.even, g.odd]:
        mask[:] = 0
        mask_rb.checkerboard(cb)
        g.set_checkerboard(mask, mask_rb)
        for mu in range(len(U)):
            st = g.eval(beta * staple(U, mu))
            markov(U[mu], st, mask)

assert abs(plaq - 0.7133781214555105) < 1e-6
