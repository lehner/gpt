#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno 2021
#
import gpt as g
import numpy as np
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

# now test with general Hermitian Fourier mass matrix
L = grid.gdimensions
scale_unitary = float(np.prod(L)) ** 0.5

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
assert abs(A0(U_mom) / r - 1) < 1e-10
eps = g.group.defect(U_mom[0])
g.message("Group defect:", eps)
assert eps < 1e-10

ngen = len(U_mom[0].otype.generators(np.complex128))
v = g(U_mom[0] * A0.m**0.5)
x = g.sum(g(g.trace(g.adj(v) * v) * 2.0 / ngen)).real / U_mom[0].grid.gsites
eps = abs(x - 1)
g.message("Compare variance:", x, eps)
assert eps < 0.01

# test distribution of A1 draw P ~ e^{-momdag inv(FT) mass FT mom}
r = A1.draw(U_mom, rng)
assert abs(A1(U_mom) / r - 1) < 1e-10

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

# do test with a GFFA choice
sqrt_mass = [[g.complex(U_mom[0].grid) for i in range(4)] for j in range(4)]
coor = g.coordinates(U_mom[0])
L = U_mom[0].grid.gdimensions
k = [coor[:, mu].astype(np.complex128) * 2.0 * np.pi / L[mu] + 1e-15 for mu in range(4)]
sin_khalf_sqr = (
    np.sin(k[0] / 2) ** 2 + np.sin(k[1] / 2) ** 2 + np.sin(k[2] / 2) ** 2 + np.sin(k[3] / 2) ** 2
)
eps2 = 0.3
M2 = 0.3
for mu in range(4):
    for nu in range(4):
        PL = (
            np.exp(-1j * k[mu] / 2)
            * np.sin(k[mu] / 2)
            * np.exp(1j * k[nu] / 2)
            * np.sin(k[nu] / 2)
            / (sin_khalf_sqr)
        )
        PT = (1.0 if mu == nu else 0.0) - PL
        sqrt_Dmunu = 1 / (sin_khalf_sqr + eps2) ** 0.5 * PT + 1 / M2**0.5 * PL
        sqrt_mass[mu][nu][:] = sqrt_Dmunu

A1 = g.qcd.scalar.action.fourier_mass_term(sqrt_mass)
A1.assert_gradient_error(rng, U_mom, U_mom, 1e-3, 1e-8)
r = A1.draw(U_mom, rng)
assert abs(A1(U_mom) / r - 1) < 1e-10

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

# Test general mass term
U = g.qcd.gauge.random(U_mom[0].grid, rng)


# first test laplacian
def __slap(dst, src):
    sU = src[0:4]
    sV = src[4:]
    dU = dst[0:4]
    dV = dst[4:]
    assert len(dst) == len(src)

    for nu in range(len(dst) - 4):
        dV[nu][:] = 0
        for mu in range(4):
            sV_p = g.cshift(sV[nu], mu, 1)
            sV_m = g.cshift(sV[nu], mu, -1)
            sU_m = g.cshift(sU[mu], mu, -1)
            dV[nu] += (
                1 / 16 * (sU[mu] * sV_p * g.adj(sU[mu]) + g.adj(sU_m) * sV_m * sU_m - 2 * sV[nu])
            )

    g.copy(dU, sU)


lap = g.qcd.gauge.algebra_laplace(U)

tmp = g.copy(U + U_mom)
tmp2 = g.copy(U + U_mom)

rng.element(tmp + tmp2)

lap(tmp, U + U_mom)
__slap(tmp2, U + U_mom)

for mu in range(8):
    eps = g.norm2(tmp[mu] - tmp2[mu])
    g.message(f"Test laplacian: {eps}")
    assert eps < 1e-10

cg = g.algorithms.inverter.block_cg({"eps": 1e-12, "maxiter": 100})
slap = g.matrix_operator(
    mat=lap, inv_mat=lap.inverse(cg), accept_list=True, accept_guess=(False, True)
)
slap2 = slap * slap


def slap2_pgrad(U, vec):
    vec_prime = g(slap * (U + vec))[-len(vec) :]
    grad1 = lap.projected_gradient(vec, U, vec_prime)
    return [g(2 * x) for x in grad1]


A2 = g.qcd.scalar.action.general_mass_term(M=slap2, sqrt_M=slap, M_projected_gradient=slap2_pgrad)

A2.assert_gradient_error(rng, U + U_mom, U + U_mom, 1e-3, 1e-8)

# test distribution
r = A2.draw(U + U_mom, rng)
eps = abs(A2(U + U_mom) / r - 1)
g.message(f"Draw test: {eps}")
assert eps < 1e-10
eps = g.group.defect(U_mom[0])
g.message("Group defect:", eps)
assert eps < 1e-10

ngen = len(U_mom[0].otype.generators(np.complex128))
x = 0.0
U_mom_prime = g(slap2 * (U + U_mom))[4:]
for mu in range(4):
    x += (
        g.sum(g(g.trace(g.adj(U_mom[mu]) * U_mom_prime[mu]) * 2.0 / ngen)).real
        / U_mom[0].grid.gsites
    )
x /= 4
eps = abs(x - 1)
g.message("Compare variance:", x, eps)
assert eps < 0.01

# coupling action
for otype, cart in [(U[0].otype, False), (U_mom[0].otype, True)]:
    A3 = g.qcd.scalar.action.coupling(omega=0.132, cartesian=cart)
    fields = [g.lattice(grid, otype) for i in range(8)]
    rng.element(fields)
    A3.assert_gradient_error(rng, fields, fields, 1e-3, 1e-8)
