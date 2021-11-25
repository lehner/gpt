#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

# load configuration
rng = g.random("test")
grid = g.grid([8, 8, 8, 16], g.double)
U = g.qcd.gauge.random(grid, rng)
V = rng.element(g.lattice(U[0]))
U_transformed = g.qcd.gauge.transformed(U, V)

# reference plaquette
P = g.qcd.gauge.plaquette(U)

# test rectangle calculation using parallel transport and copy_plan
R_1x1, R_2x1 = g.qcd.gauge.rectangle(U, [(1, 1), (2, 1)])
eps = abs(P - R_1x1)
g.message(f"Plaquette {P} versus 1x1 rectangle {R_1x1}: {eps}")
assert eps < 1e-13

# Test gauge invariance of plaquette
P_transformed = g.qcd.gauge.plaquette(U_transformed)
eps = abs(P - P_transformed)
g.message(f"Plaquette before {P} and after {P_transformed} gauge transformation: {eps}")
assert eps < 1e-13

# Test gauge invariance of R_2x1
R_2x1_transformed = g.qcd.gauge.rectangle(U_transformed, 2, 1)
eps = abs(R_2x1 - R_2x1_transformed)
g.message(
    f"R_2x1 before {R_2x1} and after {R_2x1_transformed} gauge transformation: {eps}"
)
assert eps < 1e-13

# Without trace and real projection
R_2x1_notp = g.qcd.gauge.rectangle(U_transformed, 2, 1, trace=False, real=False)
eps = abs(g.trace(R_2x1_notp).real - R_2x1)
g.message(f"R_2x1 no real and trace check: {eps}")
assert eps < 1e-13

# Test field version
R_2x1_field = g(g.sum(g.qcd.gauge.rectangle(U, 2, 1, field=True)) / U[0].grid.gsites)
eps = abs(R_2x1 - R_2x1_field)
g.message(f"R_2x1 field check: {eps}")
assert eps < 1e-13

# Without trace and real projection and field
R_2x1_notp = g.qcd.gauge.rectangle(
    U_transformed, 2, 1, trace=False, real=False, field=True
)
eps = abs(g(g.sum(g.trace(R_2x1_notp))).real / U[0].grid.gsites - R_2x1)
g.message(f"R_2x1 field, no real and trace check: {eps}")
assert eps < 1e-13

# Test clover field strength against rectangles
for mu in range(4):
    for nu in range(4):
        if mu != nu:
            Fmunu = g.qcd.gauge.field_strength(U, mu, nu)

            A, B = g.qcd.gauge.rectangle(U, [
                [ (mu,1,nu,1), (nu,-1,mu,1), (mu,-1,nu,-1), (nu,1,mu,-1) ],
                [ (nu,1,mu,1), (mu,-1,nu,1), (nu,-1,mu,-1), (mu,1,nu,-1) ]
            ], real=False, trace=False, field=True)
            Fmunutest = g(3/2*A - 3/2*B)
            eps2 = g.norm2(Fmunutest - Fmunu)
            g.message(f"F_{mu}{nu} test: {eps2}")
            assert eps2 < 1e-25
            eps2 = g.norm2(g.adj(A) - B)
            g.message(f"F_{mu}{nu} adjoint test: {eps2}")
            assert eps2 < 1e-25


# Test gauge covariance of staple
rho = np.array(
    [[0.0 if i == j else 0.1 for i in range(4)] for j in range(4)], dtype=np.float64
)
C = g.qcd.gauge.staple_sum(U, rho=rho)
C_transformed = g.qcd.gauge.staple_sum(U_transformed, rho=rho)
for mu in range(len(C)):
    q = g.sum(g.trace(C[mu] * g.adj(U[mu]))) / U[0].grid.gsites
    q_transformed = (
        g.sum(g.trace(C_transformed[mu] * g.adj(U_transformed[mu]))) / U[0].grid.gsites
    )

    eps = abs(q - q_transformed)
    g.message(
        f"Staple q[{mu}] before {q} and after {q_transformed} gauge transformation: {eps}"
    )
    assert eps < 1e-14

# Test gauge actions
for action in [g.qcd.gauge.action.wilson(5.43)]:
    action.assert_gradient_error(rng, U, U, 1e-3, 1e-8)

# Test wilson flow and energy density
U_wf = g.qcd.gauge.smear.wilson_flow(U, epsilon=0.1)
E = g.qcd.gauge.energy_density(U_wf)
E_from_field = g(
    g.sum(g.qcd.gauge.energy_density(U_wf, field=True)) / U_wf[0].grid.gsites
)
eps = abs(E - 0.3032029987236007)
g.message(f"Energy density check after wilson flow at t=0.1: {eps}")
assert eps < 1e-10
eps = abs(E - E_from_field)
g.message(f"Energy density field test: {eps}")
assert eps < 1e-10

# Test stout smearing
U_stout = U
P_stout = []
for i in range(3):
    U_stout = g.qcd.gauge.smear.stout(U_stout, rho=0.1)

    for mu in range(len(U_stout)):
        I = g.identity(U_stout[mu])
        eps2 = g.norm2(U_stout[mu] * g.adj(U_stout[mu]) - I) / g.norm2(I)
        g.message(f"Unitarity check of stout-smeared links: mu = {mu}, eps2 = {eps2}")

    P_stout.append(g.qcd.gauge.plaquette(U_stout))

g.message(f"Stout smeared plaquettes {P_stout}")
assert sorted(P_stout) == P_stout  # make sure plaquettes go towards one

# for given gauge configuration, cross-check against previous Grid code
# this establishes the randomized check value used below
# U = g.load("/hpcgpfs01/work/clehner/configs/24I_0p005/ckpoint_lat.IEEE64BIG.5000")
# P = [g.qcd.gauge.plaquette(U),g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(U, rho=0.15, orthogonal_dimension=3)),g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(U, rho=0.1))]
# P_comp = [0.588074,0.742136,0.820262]
# for i in range(3):
#    assert abs(P[i] - P_comp[i]) < 1e-5
# g.message(f"Plaquette fingerprint {P} and reference {P_comp}")

P = [
    g.qcd.gauge.plaquette(U),
    g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(U, rho=0.15, orthogonal_dimension=3)),
    g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(U, rho=0.1)),
]
P_comp = [0.7986848674527128, 0.9132213221481771, 0.9739960794712376]
g.message(f"Plaquette fingerprint {P} and reference {P_comp}")
for i in range(3):
    assert abs(P[i] - P_comp[i]) < 1e-12

# Test gauge fixing
opt = g.algorithms.optimize.non_linear_cg(maxiter=50, eps=1e-9, step=0.1)
V0 = g.identity(U[0])
rng.element(V0)

# get functionals
l = g.qcd.gauge.fix.landau(U)
fal = g.algorithms.optimize.fourier_accelerate.inverse_phat_square(V0.grid, l)

# test functionals
l.assert_gradient_error(rng, V0, V0, 1e-3, 1e-8)

# test gauge fixing
for f, f_test, tag, expected_improvement in [
    (l, l, "Landau", 1e-7),
    (fal, l, "Fourier Accelerated Landau", 1e-9),
]:
    V1 = g.copy(V0)

    eps0 = g.norm2(f_test.gradient(V1, V1)) ** 0.5 / f_test(V1)
    g.message(f"df/f before {tag} gauge fix: {eps0}")

    opt(f)([V1], [V1])

    eps1 = g.norm2(f_test.gradient(V1, V1)) ** 0.5 / f_test(V1)
    g.message(f"df/f after {tag} gauge fix: {eps1}, improvement: {eps1/eps0}")
    assert eps1 / eps0 < expected_improvement
