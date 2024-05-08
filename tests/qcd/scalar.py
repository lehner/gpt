#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno 2021
#
import gpt as g
import numpy
import sys

# grid
grid = g.grid([8, 8, 8, 8], g.double)

rng = g.random("scalar!")

phi = g.real(grid)
rng.element(phi)

actions = [g.qcd.scalar.action.mass_term(), g.qcd.scalar.action.phi4(0.119, 0.01)]

for a in actions:
    g.message(a.__name__)
    a.assert_gradient_error(rng, phi, phi, 1e-5, 1e-7)

# fourier mass term
U_mom = [g.lattice(grid, g.ot_matrix_su_n_fundamental_algebra(3)) for i in range(4)]
rng.element(U_mom)
A0 = g.qcd.scalar.action.mass_term(0.612)
sqrt_mass = [[g.complex(grid) for i in range(4)] for j in range(4)]
for mu in range(4):
    for nu in range(4):
        sqrt_mass[mu][nu][:] = 0.612**0.5 if mu == nu else 0.0

A1 = g.qcd.scalar.action.fourier_mass_term(sqrt_mass)

eps = abs(A1(U_mom) / A0(U_mom) - 1.0)
g.message(f"Regress fourier mass term: {eps}")
assert eps < 1e-10

# now test with general Hermitian mass matrix
for mu in range(4):
    for nu in range(4):
        rng.cnormal(sqrt_mass[mu][nu])
tmp = [g.copy(x) for x in sqrt_mass]
for mu in range(4):
    for nu in range(4):
        sqrt_mass[mu][nu] @= g.adj(tmp[nu][mu]) + tmp[mu][nu]

A1 = g.qcd.scalar.action.fourier_mass_term(sqrt_mass)
A1.assert_gradient_error(rng, U_mom, U_mom, 1e-3, 1e-8)
