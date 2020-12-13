#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import numpy as np

# grid
L = [8, 8, 8, 8]
grid = g.grid(L, g.single)
grid_eo = g.grid(L, g.single, g.redblack)

# hot start
g.default.push_verbose("random", False)
rng = g.random("test", "vectorized_ranlux24_24_64")  # use faster rng ; benchmark rng...
U = g.qcd.gauge.unit(grid)
Nd = len(U)

# red/black mask
mask_rb = g.complex(grid_eo)
mask_rb[:] = 1

# full mask
mask = g.complex(grid)

# identity gauge field
V_eye = g.identity(U[0])

# simple plaquette action
def staple(U, mu):
    st = g.lattice(U[0])
    st[:] = 0
    Nd = len(U)
    for nu in range(Nd):
        if mu != nu:
            st += g.qcd.gauge.staple(U, mu, nu) / U[0].otype.Nc
    return st


# metropolis loop
for it in range(2000):
    plaq = g.qcd.gauge.plaquette(U)
    beta = 5.5

    g.message(f"{it} -> {plaq}")

    number_accept = 0
    possible_accept = 0

    t = g.timer("metropolis")
    for cb in [g.even, g.odd]:
        mask[:] = 0
        mask_rb.checkerboard(cb)
        g.set_checkerboard(mask, mask_rb)

        for mu in range(Nd):

            t("staple")
            st = staple(U, mu)

            t("update")
            action = g.component.real(g.eval(-beta * g.trace(U[mu] * g.adj(st)) * mask))

            V = g.lattice(U[0])
            t("random")
            rng.element(V, scale=0.5, normal=True)
            t("update")
            V = g.where(mask, V, V_eye)

            U_mu_prime = g.eval(V * U[mu])
            action_prime = g.component.real(
                g.eval(-beta * g.trace(U_mu_prime * g.adj(st)) * mask)
            )

            dp = g.matrix.exp(action - action_prime)

            rn = g.lattice(dp)
            t("random")
            rng.uniform_real(rn)
            t("random")
            accept = dp > rn
            accept *= mask

            number_accept += g.norm2(accept)
            possible_accept += g.norm2(mask)

            U[mu] = g.where(accept, U_mu_prime, U[mu])
            t()

    g.message(t)
    g.message(f"acceptance rate: {number_accept / possible_accept}")
