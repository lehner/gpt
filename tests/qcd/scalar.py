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
L = grid.gdimensions
scale_unitary = float(numpy.prod(L)) ** 0.5

for mu in range(4):
    for nu in range(4):
        rng.normal(
            sqrt_mass[mu][nu]
        )  # start with real value and then FT makes sure we have sqrt_mass[-k] = sqrt_mass[k]^*
        sqrt_mass[mu][nu] @= scale_unitary * g.fft() * sqrt_mass[mu][nu]
tmp = [g.copy(x) for x in sqrt_mass]
for mu in range(4):
    for nu in range(4):
        sqrt_mass[mu][nu] @= g.adj(tmp[nu][mu]) + tmp[mu][nu]

A1 = g.qcd.scalar.action.fourier_mass_term(sqrt_mass)
A1.assert_gradient_error(rng, U_mom, U_mom, 1e-3, 1e-8)

# test distribution of A0 draw
r = A0.draw(U_mom, rng)
assert abs(A0(U_mom) - r) < 1e-10
eps = g.group.defect(U_mom[0])
g.message("Group defect:", eps)
assert eps < 1e-10

ngen = len(U_mom[0].otype.generators(numpy.complex128))
v = g(U_mom[0] * A0.m**0.5)
x = g.sum(g(g.trace(g.adj(v) * v) * 2.0 / ngen)).real / U_mom[0].grid.gsites
eps = abs(x - 1)
g.message("Compare variance:", x, eps)
assert eps < 0.01

# test distribution of A1 draw P ~ e^{-momdag inv(FT) mass FT mom}
r = A1.draw(U_mom, rng)
assert abs(A1(U_mom) - r) < 1e-10

# group defect would be triggered if sqrt_mass does not have sqrt_mass[k] = sqrt_mass[-k]
eps = g.group.defect(U_mom[0])
g.message("Group defect:", eps)
assert eps < 1e-10

ft_mom = g(scale_unitary * g.fft() * U_mom)

eps = abs(g.norm2(U_mom[0]) / g.norm2(ft_mom[0]) - 1)
g.message("Unitary FT:", eps)
assert eps < 1e-10

# make them uniformly gaussian
v = g.lattice(U_mom[0])
for mu in range(4):
    v[:] = 0
    for nu in range(4):
        v += sqrt_mass[mu][nu] * ft_mom[nu]
    x = g.sum(g(g.trace(g.adj(v) * v) * 2.0 / ngen)).real / U_mom[0].grid.gsites
    eps = abs(x - 1)
    g.message("Compare variance:", x, eps)
    assert eps < 0.01
