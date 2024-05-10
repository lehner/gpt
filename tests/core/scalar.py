#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g

m0crit = 4.0
m0 = g.default.get_float("--mass", 0.1) + m0crit

grid = g.grid([4, 4, 4, 4], g.double)

src = g.complex(grid)
src[:] = 0
src[0, 0, 0, 0] = 1


# Create a free Klein-Gordon operator (spectrum from mass^2-16 .. mass^2)
def A(dst, src, mass):
    assert dst != src
    dst @= (mass**2.0) * src
    for i in range(4):
        dst += g.cshift(src, i, 1) + g.cshift(src, i, -1) - 2 * src


# Matrix
M = g.matrix_operator(lambda d, s: A(d, s, m0))

# find largest eigenvalue
powit = g.algorithms.eigen.power_iteration({"eps": 1e-6, "maxiter": 100})
g.message("Largest eigenvalue: ", powit(M, src)[0])

# perform CG
psi = g.lattice(src)
psi[:] = 0

cg = g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 1000})

psi @= cg(M) * src

# Test CG
eps2 = g.norm2(M * psi - src)
g.message("True residuum:", eps2)
assert eps2 < 1e-25
