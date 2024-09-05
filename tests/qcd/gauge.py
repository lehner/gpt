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

# quadruple precision global sum version
U_quad = g.convert(U, g.double_quadruple)

# reference plaquette
P = g.qcd.gauge.plaquette(U)

# invariant distance (test axioms)
ds = g.group.invariant_distance(U[0], U[1])
dsp = g.group.invariant_distance(g.group.compose(V, U[0]), g.group.compose(V, U[1]))
assert abs(ds / dsp - 1) < 1e-12

ds = g.group.invariant_distance(U[0], U[1])
dsp = g.group.invariant_distance(g.group.compose(U[0], V), g.group.compose(U[1], V))
assert abs(ds / dsp - 1) < 1e-12

ds = g.group.invariant_distance(U[0], U[0])
assert ds < 1e-12

ds = g.group.invariant_distance(U[0], U[1])
dsp = g.group.invariant_distance(U[1], U[0])
assert abs(ds / dsp - 1) < 1e-12

mom = [g.group.cartesian(u) for u in U]
rng.element(mom)
ds = g.group.invariant_distance(mom[0], mom[1])
dsp = g.group.invariant_distance(g.group.compose(mom[2], mom[0]), g.group.compose(mom[2], mom[1]))
assert abs(ds / dsp - 1) < 1e-12

# import cgpt, sys

# smr = []
# for mu in range(4):
#     for cb in [g.even, g.odd]:
#         smr.append(g.qcd.gauge.smear.local_stout(rho=0.124, dimension=mu, checkerboard=cb))

# smrr = list(reversed(smr))
# act = smrr[0].action_log_det_jacobian()
# for s in smrr[1:]:
#     act = act.transformed(s) + s.action_log_det_jacobian()

# Usm = U
# for s in smr:
#     Usm = s(Usm)
# g.message(P, g.qcd.gauge.plaquette(Usm), act(U))

# cgpt.test_grid(U + Usm + [g(1j*x) for x in act.gradient(U, U)])

# act.assert_gradient_error(rng, U, U, 1e-3, 1e-8)

# sys.exit(0)


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
g.message(f"R_2x1 before {R_2x1} and after {R_2x1_transformed} gauge transformation: {eps}")
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
R_2x1_notp = g.qcd.gauge.rectangle(U_transformed, 2, 1, trace=False, real=False, field=True)
eps = abs(g(g.sum(g.trace(R_2x1_notp))).real / U[0].grid.gsites - R_2x1)
g.message(f"R_2x1 field, no real and trace check: {eps}")
assert eps < 1e-13

# Test clover field strength against rectangles
for mu in range(4):
    for nu in range(4):
        if mu != nu:
            Fmunu = g.qcd.gauge.field_strength(U, mu, nu)

            A, B = g.qcd.gauge.rectangle(
                U,
                [
                    [
                        (mu, 1, nu, 1),
                        (nu, -1, mu, 1),
                        (mu, -1, nu, -1),
                        (nu, 1, mu, -1),
                    ],
                    [
                        (nu, 1, mu, 1),
                        (mu, -1, nu, 1),
                        (nu, -1, mu, -1),
                        (mu, 1, nu, -1),
                    ],
                ],
                real=False,
                trace=False,
                field=True,
            )
            Fmunutest = g(3 / 2 * A - 3 / 2 * B)
            eps2 = g.norm2(Fmunutest - Fmunu)
            g.message(f"F_{mu}{nu} test: {eps2}")
            assert eps2 < 1e-25
            eps2 = g.norm2(g.adj(A) - B)
            g.message(f"F_{mu}{nu} adjoint test: {eps2}")
            assert eps2 < 1e-25


# Test gauge covariance of staple
rho = np.array([[0.0 if i == j else 0.1 for i in range(4)] for j in range(4)], dtype=np.float64)
C = g.qcd.gauge.staple_sum(U, rho=rho)
C_transformed = g.qcd.gauge.staple_sum(U_transformed, rho=rho)
for mu in range(len(C)):
    q = g.sum(g.trace(C[mu] * g.adj(U[mu]))) / U[0].grid.gsites
    q_transformed = g.sum(g.trace(C_transformed[mu] * g.adj(U_transformed[mu]))) / U[0].grid.gsites

    eps = abs(q - q_transformed)
    g.message(f"Staple q[{mu}] before {q} and after {q_transformed} gauge transformation: {eps}")
    assert eps < 1e-14

# Test topology
Q = g.qcd.gauge.topological_charge(U)
eps = abs(Q - 0.18736242691275048)
g.message(f"Test field_strength Q definition: {eps}")
assert eps < 1e-13

g.message("Test diff top")
adU = [g.ad.reverse.node(g.copy(u)) for u in U]
dQ = g.qcd.gauge.differentiable_topology(adU)
diff_Q = dQ.functional(*adU)
diff_Q.assert_gradient_error(rng, U, U, 1e-3, 1e-8)
assert abs(Q - diff_Q(U)) < 1e-13

Q = g.qcd.gauge.topological_charge_5LI(U, cache={})
eps = abs(Q - 0.32270083147744544)
g.message(f"Test 5LI Q definition: {eps}")
assert eps < 1e-13

# Test differentiable energy_density
g.message("Test diff E")
E = g.qcd.gauge.energy_density(U)
dE = g.qcd.gauge.differentiable_energy_density(adU)
diff_E = dE.functional(*adU)
diff_E.assert_gradient_error(rng, U, U, 1e-3, 1e-8)
assert abs(E - diff_E(U)) < 1e-13

# Test gauge actions
for action in [g.qcd.gauge.action.wilson(5.43), g.qcd.gauge.action.iwasaki(5.41)]:
    # test action double precision versus quadruple precision
    a_ref = action(U)
    a_quad = action(U_quad)
    eps = abs((float(a_quad) - a_ref) / a_ref)
    g.message(f"Action {action.__name__} quad precision regression against double precision: {eps}")
    assert eps < 1e-14

    # test original action gradient
    action.assert_gradient_error(rng, U, U, 1e-3, 1e-8)

    # test stout smearing chain rule
    sm = g.qcd.gauge.smear.stout(rho=0.136)
    action_sm = action.transformed(sm)
    action_sm.assert_gradient_error(rng, U, U, 1e-3, 1e-7)

    # test local (factorizable) stout smearing
    lsm = g.qcd.gauge.smear.local_stout(rho=0.05, dimension=1, checkerboard=g.even)
    action_sm = action.transformed(lsm)
    action_sm.assert_gradient_error(rng, U, U, 1e-3, 1e-7)
    lsm.assert_log_det_jacobian(U, 1e-5, (2, 2, 2, 0), 1e-7)

    action_log_det = lsm.action_log_det_jacobian()
    action_log_det.assert_gradient_error(rng, U, U, 1e-3, 1e-8)

    st = action.staples(U)
    for mu in range(len(U)):
        adj_staple = g(g.adj(st[mu]))
        Uprime = g.copy(U)
        Uprime[mu][0, 1, 2, 3] *= 1.1
        action_diff = action(U) - action(Uprime)
        action_a = -g.sum(g.trace(U[mu] * adj_staple)).real
        action_b = -g.sum(g.trace(Uprime[mu] * adj_staple)).real
        eps = abs(action_a - action_b - action_diff) / U[0].grid.gsites
        g.message(
            f"Action {action.__name__} staple representation in U_{mu} variation accurate to {eps}"
        )
        assert eps < 1e-13

# test instantiation of additional short-hands
g.qcd.gauge.action.dbw2(4.3)
g.qcd.gauge.action.symanzik(4.3)

# Test wilson flow and energy density
U_wf = g.qcd.gauge.smear.wilson_flow(U, epsilon=0.1)
E = g.qcd.gauge.energy_density(U_wf)
E_from_field = g(g.sum(g.qcd.gauge.energy_density(U_wf, field=True)) / U_wf[0].grid.gsites)
eps = abs(E - 0.3032029987236007)
g.message(f"Energy density check after wilson flow at t=0.1: {eps}")
assert eps < 1e-10
eps = abs(E - E_from_field)
g.message(f"Energy density field test: {eps}")
assert eps < 1e-10

# Test stout smearing
U_stout = U
P_stout = []
sm = g.qcd.gauge.smear.stout(rho=0.1)
for i in range(3):
    U_stout = sm(U_stout)

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
# P = [g.qcd.gauge.plaquette(U),g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(rho=0.15, orthogonal_dimension=3)(U)),g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(rho=0.1)(U))]
# P_comp = [0.588074,0.742136,0.820262]
# for i in range(3):
#    assert abs(P[i] - P_comp[i]) < 1e-5
# g.message(f"Plaquette fingerprint {P} and reference {P_comp}")

P = [
    g.qcd.gauge.plaquette(U),
    g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(rho=0.15, orthogonal_dimension=3)(U)),
    g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(rho=0.1)(U)),
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
l.assert_gradient_error(rng, V0, V0, 1e-3, 1e-7)

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
    g.message(f"df/f after {tag} gauge fix: {eps1}, improvement: {eps1 / eps0}")
    assert eps1 / eps0 < expected_improvement


# test temporal gauge
V = g.qcd.gauge.fix.identity(U, mu=3)

Up = g.qcd.gauge.transformed(U, V)

ref = g.identity(V)[0, 0, 0, 0]

for t in range(grid.gdimensions[3] - 1):
    eps2 = g.norm2(Up[3][0, 0, 0, t] - ref)
    g.message(f"Test temporal gauge at t={t}: {eps2}")
    assert eps2 < 1e-25

# test reversibility of local_stout
for rho in [0.05, 0.1, 0.124, 0.25]:
    for mu in range(4):
        for cb in [g.even, g.odd]:
            g.message(f"Testing reversibility for rho={rho}, mu={mu}, cb={cb.__name__}")
            lsm = g.qcd.gauge.smear.local_stout(rho=rho, dimension=mu, checkerboard=cb)
            Uprime = lsm(U)
            U0 = lsm.inv(Uprime)
            if U0 is None:
                assert rho > 1 / 8
            else:
                eps2 = 0.0
                for nu in range(4):
                    eps2 += g.norm2(U[nu] - U0[nu]) / g.norm2(U0[nu])
                g.message(eps2)
                assert eps2 < 1e-28


# test general differentiable field transformation framework
ft_stout = g.qcd.gauge.smear.differentiable_stout(rho=0.01)

fr = g.algorithms.optimize.fletcher_reeves
ls2 = g.algorithms.optimize.line_search_quadratic

dft = g.qcd.gauge.smear.differentiable_field_transformation(
    U,
    ft_stout,
    # g.algorithms.inverter.fgmres(eps=1e-15, maxiter=1000, restartlen=60),
    # g.algorithms.inverter.fgmres(eps=1e-15, maxiter=1000, restartlen=60),
    g.algorithms.inverter.fgcr(eps=1e-13, maxiter=1000, restartlen=60),
    g.algorithms.inverter.fgcr(eps=1e-13, maxiter=1000, restartlen=60),
    g.algorithms.optimize.non_linear_cg(
        maxiter=1000, eps=1e-15, step=1e-1, line_search=ls2, beta=fr
    ),
)

dfm = dft.diffeomorphism()
ald = dft.action_log_det_jacobian()

# test diffeomorphism of stout against reference implementation
dfm_ref = g.qcd.gauge.smear.stout(rho=0.01)
Uft = dfm(U)
Uft_ref = dfm_ref(U)
for mu in range(4):
    eps2 = g.norm2(Uft[mu] - Uft_ref[mu]) / g.norm2(Uft[mu])
    g.message("Test ft:", eps2)
    assert eps2 < 1e-25

mom = [g.group.cartesian(u) for u in U]
mom_prime = g.copy(mom)
rng.normal_element(mom_prime)
t0 = g.time()
mom = dfm.jacobian(U, Uft, mom_prime)
t1 = g.time()
mom_ref = dfm_ref.jacobian(U, Uft, mom_prime)
t2 = g.time()
for mu in range(4):
    eps2 = g.norm2(mom[mu] - mom_ref[mu]) / g.norm2(mom[mu])
    g.message("Test jacobian:", eps2)
    assert eps2 < 1e-25

g.message("Time for dfm.jacobian", t1 - t0, "seconds")
g.message("Time for dfm_ref.jacobian", t2 - t1, "seconds")

mom2 = g.copy(mom)
g.message("Action log det jac:", ald(U + mom2))

ald.assert_gradient_error(rng, U + mom2, U, 1e-3, 1e-7)

act = ald.draw(U + mom2, rng)
act2 = ald(U + mom2)
eps = abs(act / act2 - 1)
g.message("Draw from log det action:", eps)
assert eps < 1e-8

if True:
    U0 = dft.inverse(Uft)
    for mu in range(4):
        eps2 = g.norm2(U0[mu] - U[mu]) / g.norm2(U[mu])
        g.message("Test invertibility:", eps2)
        assert eps2 < 1e-25
