#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys, cmath

# load configuration
rng = g.random("test")
L = [8, 8, 8, 16]
U = g.qcd.gauge.random(g.grid(L, g.double), rng)

# do everything in single-precision
U = g.convert(U, g.single)

# plaquette
g.message("Plaquette:", g.qcd.gauge.plaquette(U))

# use the gauge configuration grid
grid = U[0].grid

# wilson parameters
p = {
    "kappa": 0.137,
    "csw_r": 0.0,
    "csw_t": 0.0,
    "xi_0": 1.0,
    "nu": 1.0,
    "isAnisotropic": False,
    "boundary_phases": [cmath.exp(1j), cmath.exp(2j), cmath.exp(3j), cmath.exp(4j)],
}

# pf=g.params("~/gpt/tests/wilson.txt")
# print(pf)

# take slow reference implementation of wilson (kappa = 1/2/(m0 + 4) ) ;
w_ref = g.qcd.fermion.reference.wilson_clover(U, p)

# and fast Grid version
w = g.qcd.fermion.wilson_clover(U, p, kappa=0.137)

# create point source
src = rng.cnormal(g.vspincolor(grid))

dst_ref, dst = g.lattice(src), g.lattice(src)

# correctness
dst_ref @= w_ref * src
dst @= w * src

eps = g.norm2(dst - dst_ref) / g.norm2(dst)
g.message("Test wilson versus reference:", eps)
assert eps < 1e-13

# now timing
t0 = g.time()
for i in range(100):
    w_ref(dst_ref, src)
t1 = g.time()
for i in range(100):
    w(dst, src)
t2 = g.time()
for i in range(100):
    dst = w(src)
t3 = g.time()
for i in range(100):
    dst @= w * src
t4 = g.time()

g.message("Reference time/s: ", t1 - t0)
g.message("Grid time/s (reuse lattices): ", t2 - t1)
g.message("Grid time/s (with temporaries): ", t3 - t2)
g.message("Grid time/s (with expressions): ", t4 - t3)

# manual test of twisted mass term
tm_test_wilson = g.qcd.fermion.wilson_clover(
    U,
    mass=-1.8,
    csw_r=0,
    csw_t=0,
    cF=1.0,
    xi_0=1.0,
    nu=1,
    isAnisotropic=True,
    boundary_phases=[-1, 1, 1, -1],
)
tm_test_wilson_tm = g.qcd.fermion.wilson_twisted_mass(
    U, mass=-1.8, mu=0.3, boundary_phases=[-1, 1, 1, -1]
)
eps2 = g.norm2(tm_test_wilson * src + 1j * 0.3 * g.gamma[5] * src - tm_test_wilson_tm * src)
g.message(f"Test of twisted mass term: {eps2}")
assert eps2 < 1e-7

# create point source
src = g.mspincolor(grid)
g.create.point(src, [1, 0, 0, 0])  # pick point 1 so that "S" in preconditioner contributes to test

# build solver using g5m and cg
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-6, "maxiter": 1000})

slv = w.propagator(inv.preconditioned(pc.g5m_ne(), cg))
slv_eo1 = w.propagator(inv.preconditioned(pc.eo1_ne(), cg))
slv_eo2 = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

# propagator
dst_eo1 = g.mspincolor(grid)
dst_eo2 = g.mspincolor(grid)

dst_eo1 @= slv_eo1 * src
iter_eo1 = len(cg.history)

dst_eo2 @= slv_eo2 * src
iter_eo2 = len(cg.history)

eps2 = g.norm2(dst_eo1 - dst_eo2) / g.norm2(dst_eo1)
g.message(
    f"Result of test EO1 versus EO2 preconditioning: eps2={eps2} iter1={iter_eo1} iter2={iter_eo2}"
)
assert eps2 < 1e-12

# true residuum
eps2 = g.norm2(w * dst_eo1 - src) / g.norm2(src)
g.message("Result of M M^-1 = 1 test: eps2=", eps2)
assert eps2 < 1e-10

# and a reference
if True:
    dst = g.mspincolor(grid)
    dst @= slv * src
    eps2 = g.norm2(dst_eo1 - dst) / g.norm2(dst_eo1)
    g.message("Result of test EO1 versus G5M: eps2=", eps2)
    assert eps2 < 1e-10

dst = dst_eo2

# invert daggered
v1 = rng.cnormal(g.vspincolor(grid))
v2 = rng.cnormal(g.vspincolor(grid))
w_v1 = g(inv.preconditioned(pc.eo1_ne(), cg)(w) * v1)
wadj_v2 = g(inv.preconditioned(pc.eo1_ne(), cg)(w.adj()) * v2)

eps = abs(g.inner_product(v2, w_v1) / g.inner_product(v1, wadj_v2).conjugate() - 1.0)
g.message(f"Test inv(dag(w)): {eps}")
assert eps < 1e-5

# two-point
correlator = g.slice(g.trace(dst * g.adj(dst)), 3)

# test value of correlator
correlator_ref = [
    1.0710210800170898,
    0.08988216519355774,
    0.015699388459324837,
    0.003721018321812153,
    0.0010877142194658518,
    0.0003579717595130205,
    0.00012700144725386053,
    5.180457083042711e-05,
    3.406393443583511e-05,
    5.2738148951902986e-05,
    0.0001297977869398892,
    0.0003634534077718854,
    0.0011047901352867484,
    0.0037904218770563602,
    0.015902264043688774,
    0.09077762067317963,
]

# output
for t, c in enumerate(correlator):
    g.message(t, c.real, correlator_ref[t])

eps = np.linalg.norm(np.array(correlator) - np.array(correlator_ref))
g.message("Expected correlator eps: ", eps)
assert eps < 1e-5


# test conserved current
def divergence(f, current):
    resN = g.lattice(f)
    resN[:] = 0

    for mu in range(4):
        c_mu = current(f, f, mu)
        resN += c_mu - g.cshift(c_mu, mu, -1)

    return g.norm2(resN)


def local_current(f, b, mu):
    return g(g.gamma[mu] * g.gamma[5] * g.adj(f) * g.gamma[5] * b)


div_J = divergence(dst, w.conserved_vector_current)
assert div_J < 1e-13
g.message(f"Divergence of conserved Wilson current {div_J}")

div_J = divergence(dst, local_current)
g.message(f"Divergence of local current {div_J}")

# split grid solver check
slv_split_eo1 = w.propagator(
    inv.preconditioned(
        pc.eo1_ne(), inv.split(cg, mpi_split=g.default.get_ivec("--mpi_split", None, 4))
    )
)
dst_split = g.mspincolor(grid)
dst_split @= slv_split_eo1 * src
eps2 = g.norm2(dst_split - dst_eo1) / g.norm2(dst_eo1)
g.message(f"Split grid solver check {eps2}")
assert eps2 < 1e-12


# gauge transformation check
V = rng.element(g.mcolor(grid))
prop_on_transformed_U = w.updated(g.qcd.gauge.transformed(U, V)).propagator(
    inv.preconditioned(pc.eo2_ne(), cg)
)
prop_transformed = g.qcd.gauge.transformed(slv_eo2, V)
src = rng.cnormal(g.vspincolor(grid))
dst1 = g(prop_on_transformed_U * src)
dst2 = g(prop_transformed * src)
eps2 = g.norm2(dst1 - dst2) / g.norm2(dst1)
g.message(f"Gauge transformation check {eps2}")
assert eps2 < 1e-12


# test twisted boundary momentum phase
U_unit = g.qcd.gauge.unit(grid)
theta = 0.91231
quark0 = g.qcd.fermion.mobius(
    U_unit,
    Ls=8,
    mass=0.1,
    b=1,
    c=0,
    M5=1.8,
    boundary_phases=[np.exp(1j * theta), 1, 1, 1],
)
q0prop = quark0.propagator(inv.preconditioned(pc.eo2_ne(), inv.cg(eps=1e-7, maxiter=1000)))
src = g.vspincolor(U_unit[0].grid)
src[:] = 0
src[:, :, :, 0, 0, 0] = 1
prop = g(q0prop * g.exp_ixp(np.array([theta / L[0], 0, 0, 0])) * src)
prop_1000_over_0000 = complex(prop[1, 0, 0, 2, 0, 0]) / complex(prop[0, 0, 0, 2, 0, 0])
eps = abs(prop_1000_over_0000) - 1
g.message(f"Twisted boundary covariance is a phase: {eps}")
assert eps < 1e-6
eps = abs(prop_1000_over_0000 - np.exp(1j * theta / L[0]))
g.message(f"Twisted boundary covariance as expected momentum: {eps}")
assert eps < 1e-6


# test instantiation of other actions
rhq = g.qcd.fermion.rhq_columbia(U, mass=4.0, cp=3.0, zeta=2.5, boundary_phases=[1, 1, 1, -1])


#########################################################################
# Test fermion operators against known results
#
# this protects against bugs introduced in the matrix application
# of fermion operators (new architectures / implementations)
#
# These tests are only meaningfull if reference implementation is also
# included
#########################################################################
wilson_params = {
    "kappa": 0.13500,
    "csw_r": 0.0,
    "csw_t": 0.0,
    "xi_0": 1.33111,
    "nu": 2.61,
    "isAnisotropic": True,
    "boundary_phases": [1.0, -1.0, 1.0, -1.0],
}
wilson_clover_params = {
    "kappa": 0.13500,
    "csw_r": 1.5,
    "csw_t": 1.951,
    "xi_0": 1.33111,
    "nu": 2.61,
    "isAnisotropic": True,
    "boundary_phases": [1.0, -1.0, 1.0, -1.0],
}
wilson_clover_open_params = {
    "kappa": 0.13500,
    "csw_r": 1.978,
    "csw_t": 1.978,
    "cF": 1.3,
    "xi_0": 1,
    "nu": 1,
    "isAnisotropic": False,
    "boundary_phases": [1.0, 1.0, 1.0, 0.0],
}
wilson_twisted_mass_params = {
    "mass": -1.8,
    "mu": 0.2,
    "boundary_phases": [1.0, 1.0, 1.0, -1.0],
}
wilson_clover_matrices = {
    "": [(-946.8714968698364 - 427.1253034080037j)],
    ".Mdiag": [(-908.620454398646 - 3428.779878527792j)],
}
wilson_matrices = {
    "": [(-999.7564252326631 - 466.7758727463097j)],
    ".Mdiag": [(-961.5053827614738 - 3468.430447866095j)],
}
wilson_clover_matrices_open = {
    "": [(-1634.2615676797234 + 239.27037187495998j)],
    ".Mdiag": [(-1239.3535155227526 - 1158.5295177146759j)],
}
wilson_twisted_mass_matrices = {
    "": [(-5.665095757463064 + 373.96051873176737j)],
    ".Mdiag": [(-440.5312395819657 - 1102.362512575698j)],
}
test_suite = {
    "zmobius": {
        "fermion": g.qcd.fermion.zmobius,
        "params": {
            "mass": 0.08,
            "M5": 1.8,
            "b": 1.0,
            "c": 0.0,
            "omega": [
                0.17661651536320583 + 1j * (0.14907774771612217),
                0.23027432016909377 + 1j * (-0.03530801572584271),
                0.3368765581549033 + 1j * (0),
                0.7305711010541054 + 1j * (0),
                1.1686138337986505 + 1j * (0.3506492418109086),
                1.1686138337986505 + 1j * (-0.3506492418109086),
                0.994175013717952 + 1j * (0),
                0.5029903152251229 + 1j * (0),
                0.23027432016909377 + 1j * (0.03530801572584271),
                0.17661651536320583 + 1j * (-0.14907774771612217),
            ],
            "boundary_phases": [1.0, 1.0, 1.0, -1.0],
        },
        "matrices": {
            "": [(-2424.048033434305 + 10557.661684178218j)],
            ".Mdiag": [(2643.396577965267 + 6550.259431381319j)],
            ".ImportPhysicalFermionSource": [(4064.7879718582053 - 1357.0856808000196j)],
        },
    },
    "mobius": {
        "fermion": g.qcd.fermion.mobius,
        "params": {
            "mass": 0.08,
            "M5": 1.8,
            "b": 1.5,
            "c": 0.5,
            "Ls": 12,
            "boundary_phases": [1.0, -1.0, 1.0, -1.0],
        },
        "matrices": {
            "": [(-8693.09425573421 - 4130.7793316734915j)],
            ".Mdiag": [(-4966.960264746144 - 2525.83968136146j)],
            ".ImportPhysicalFermionSource": [(-97.93443075273976 - 690.6405168964976j)],
        },
    },
    "mobius_axial_mass": {
        "fermion": g.qcd.fermion.mobius,
        "params": {
            "mass_plus": 0.08,
            "mass_minus": 0.11,
            "M5": 1.8,
            "b": 1.5,
            "c": 0.5,
            "Ls": 12,
            "boundary_phases": [1.0, -1.0, 1.0, -1.0],
        },
        "matrices": {
            "": [(-8690.547330400455 - 4127.148886222195j)],
            ".Mdiag": [(-4967.102993398692 - 2525.589904941078j)],
        },
    },
    "wilson": {
        "fermion": g.qcd.fermion.wilson_clover,
        "params": wilson_params,
        "matrices": wilson_matrices,
    },
    "wilson_clover": {
        "fermion": g.qcd.fermion.wilson_clover,
        "params": wilson_clover_params,
        "matrices": wilson_clover_matrices,
    },
    "wilson_clover_legacy": {
        "fermion": g.qcd.fermion.wilson_clover,
        "params": {
            **wilson_clover_params,
            "use_legacy": True,
        },  # will be deprecated eventually
        "matrices": wilson_clover_matrices,
    },
    "wilson_clover_reference": {
        "fermion": g.qcd.fermion.reference.wilson_clover,
        "params": wilson_clover_params,
        "matrices": wilson_clover_matrices,
    },
    "wilson_clover_openbc_reference": {
        "fermion": g.qcd.fermion.reference.wilson_clover,
        "params": wilson_clover_open_params,
        "matrices": wilson_clover_matrices_open,
    },
    "wilson_clover_openbc": {
        "fermion": g.qcd.fermion.wilson_clover,
        "params": wilson_clover_open_params,
        "matrices": wilson_clover_matrices_open,
    },
    "wilson_twisted_mass": {
        "fermion": g.qcd.fermion.wilson_twisted_mass,
        "params": wilson_twisted_mass_params,
        "matrices": wilson_twisted_mass_matrices,
    },
}


finger_print_tolerance = 100.0


def verify_single_versus_double_precision(rng, fermion_dp, fermion_sp):
    eps_ref = fermion_sp.F_grid.precision.eps * finger_print_tolerance
    for atag in fermion_dp.__dict__.keys():
        a_dp = getattr(fermion_dp, atag)
        if isinstance(a_dp, g.projected_matrix_operator):
            a_sp = getattr(fermion_sp, atag)
            rhs_dp = rng.cnormal(g.lattice(a_dp.grid[1], a_dp.otype[1]))
            lhs_dp = rng.cnormal(g.lattice(a_dp.grid[0], a_dp.otype[0]))
            if rhs_dp.grid.cb.n == 1:
                parities = [(g.full, g.full)]
            elif a_dp.parity == g.even:
                parities = [(g.even, g.even), (g.odd, g.odd)]
            elif a_dp.parity == g.odd:
                parities = [(g.odd, g.even), (g.even, g.odd)]
            else:
                assert False
            for lp, rp in parities:
                if lp != g.full:
                    rhs_dp.checkerboard(rp)
                    lhs_dp.checkerboard(lp)
                rhs_sp = g.convert(rhs_dp, g.single)
                lhs_sp = g.convert(lhs_dp, g.single)
                # first test matrix
                ref_list = a_dp(lhs_dp, rhs_dp)
                cmp_list = g.convert(a_sp(lhs_sp, rhs_sp), g.double)
                for r, c in zip(ref_list, cmp_list):
                    eps = g.norm2(r - c) ** 0.5 / g.norm2(r) ** 0.5
                    g.message(f"Verify single <> double for {atag}: {eps}")
                    assert eps < eps_ref
                # then test adjoint matrix
                ref_list = a_dp.adj()(rhs_dp, lhs_dp)
                cmp_list = g.convert(a_sp.adj()(rhs_sp, lhs_sp), g.double)
                for r, c in zip(ref_list, cmp_list):
                    eps = g.norm2(r - c) ** 0.5 / g.norm2(r) ** 0.5
                    g.message(f"Verify single <> double for {atag}.adj(): {eps}")
                    assert eps < eps_ref
        elif isinstance(a_dp, g.matrix_operator):
            a_sp = getattr(fermion_sp, atag)
            rhs_dp = rng.cnormal(a_dp.vector_space[1].lattice())
            lhs_dp = rng.cnormal(a_dp.vector_space[0].lattice())
            if rhs_dp.grid.cb.n != 1:
                # for now test only odd cb
                rhs_dp.checkerboard(g.odd)
                lhs_dp.checkerboard(g.odd)
            rhs_sp = g.convert(rhs_dp, g.single)
            lhs_sp = g.convert(lhs_dp, g.single)
            # first test matrix
            ref = a_dp(rhs_dp)
            eps = g.norm2(ref - g.convert(a_sp(rhs_sp), g.double)) ** 0.5 / g.norm2(ref) ** 0.5
            g.message(f"Verify single <> double for {atag}: {eps}")
            assert eps < eps_ref
            # then test adjoint matrix
            if a_dp.adj_mat is not None:
                ref = a_dp.adj()(lhs_dp)
                eps = (
                    g.norm2(ref - g.convert(a_sp.adj()(lhs_sp), g.double)) ** 0.5
                    / g.norm2(ref) ** 0.5
                )
                g.message(f"Verify single <> double for {atag}.adj(): {eps}")
                assert eps < eps_ref
                if a_dp.inv_mat is not None:
                    ref = a_dp.inv()(lhs_dp)
                    eps = (
                        g.norm2(ref - g.convert(a_sp.inv()(lhs_sp), g.double)) ** 0.5
                        / g.norm2(ref) ** 0.5
                    )
                    g.message(f"Verify single <> double for {atag}.inv(): {eps}")
                    assert eps < eps_ref
                    ref = a_dp.adj().inv()(lhs_dp)
                    eps = (
                        g.norm2(ref - g.convert(a_sp.adj().inv()(lhs_sp), g.double)) ** 0.5
                        / g.norm2(ref) ** 0.5
                    )
                    g.message(f"Verify single <> double for {atag}.adj().inv(): {eps}")
                    assert eps < eps_ref


def verify_daggered(rng, fermion, fermion_daggered):
    eps_ref = fermion.F_grid.precision.eps * finger_print_tolerance
    for atag in fermion.__dict__.keys():
        a = getattr(fermion, atag)
        if isinstance(a, g.projected_matrix_operator):
            a_daggered = getattr(fermion_daggered, atag)
            rhs = rng.cnormal(g.lattice(a.grid[1], a.otype[1]))
            lhs = rng.cnormal(g.lattice(a.grid[0], a.otype[0]))
            if rhs.grid.cb.n == 1:
                parities = [(g.full, g.full)]
            elif a.parity == g.even:
                parities = [(g.even, g.even), (g.odd, g.odd)]
            elif a.parity == g.odd:
                parities = [(g.odd, g.even), (g.even, g.odd)]
            else:
                assert False
            for lp, rp in parities:
                if lp != g.full:
                    rhs.checkerboard(rp)
                    lhs.checkerboard(lp)
                # first test matrix
                ref_list = a(lhs, rhs)
                cmp_list = a_daggered.adj()(lhs, rhs)
                for r, c in zip(ref_list, cmp_list):
                    eps = g.norm2(r - c) ** 0.5 / g.norm2(r) ** 0.5
                    g.message(f"Verify operator <> daggered for {atag}: {eps}")
                    assert eps < eps_ref
                # then test adjoint matrix
                ref_list = a.adj()(rhs, lhs)
                cmp_list = a_daggered(rhs, lhs)
                for r, c in zip(ref_list, cmp_list):
                    eps = g.norm2(r - c) ** 0.5 / g.norm2(r) ** 0.5
                    g.message(f"Verify operator <> daggered for {atag}.adj(): {eps}")
                    assert eps < eps_ref

        elif isinstance(a, g.matrix_operator) and a.adj_mat is not None:
            a_daggered = getattr(fermion_daggered, atag)
            rhs = rng.cnormal(a.vector_space[1].lattice())
            lhs = rng.cnormal(a.vector_space[0].lattice())
            if rhs.grid.cb.n != 1:
                # for now test only odd cb
                rhs.checkerboard(g.odd)
                lhs.checkerboard(g.odd)

            # first test matrix
            ref = a(rhs)
            eps = g.norm2(ref - a_daggered.adj()(rhs)) ** 0.5 / g.norm2(ref) ** 0.5
            g.message(f"Verify operator <> daggered for {atag}: {eps}")
            assert eps < eps_ref
            ref = a.adj()(lhs)
            eps = g.norm2(ref - a_daggered(lhs)) ** 0.5 / g.norm2(ref) ** 0.5
            g.message(f"Verify operator <> daggered for {atag}.adj(): {eps}")
            assert eps < eps_ref
            if a.inv_mat is not None:
                ref = a.inv()(lhs)
                eps = g.norm2(ref - a_daggered.adj().inv()(lhs)) ** 0.5 / g.norm2(ref) ** 0.5
                g.message(f"Verify operator <> daggered for {atag}.inv(): {eps}")
                assert eps < eps_ref
                ref = a.adj().inv()(lhs)
                eps = g.norm2(ref - a_daggered.inv()(lhs)) ** 0.5 / g.norm2(ref) ** 0.5
                g.message(f"Verify operator <> daggered for {atag}.adj().inv(): {eps}")
                assert eps < eps_ref


def verify_projected_even_odd(M, Meo, dst_p, src_p, src):
    src_proj_p = g.lattice(src)
    src_proj_p[:] = 0
    g.set_checkerboard(src_proj_p, src_p)

    dst_proj_p = g.eval(M * src_proj_p)
    g.pick_checkerboard(dst_p.checkerboard(), dst_p, dst_proj_p)

    reference = g.copy(dst_p)

    dst_p @= Meo * src_p

    eps = (g.norm2(dst_p - reference) / g.norm2(reference)) ** 0.5
    eps_ref = src.grid.precision.eps * finger_print_tolerance
    g.message(f"Test projected full matrix result versus eo matrix: {eps}")
    assert eps < eps_ref


def get_matrix(f, t):
    if t == "":
        return f
    return getattr(f, t.replace(".", ""))


def verify_matrix_element(fermion, dst, src, tag):
    mat = get_matrix(fermion_dp, tag)
    if isinstance(src, dict):
        src = src[mat.vector_space[1].grid]
        dst = dst[mat.vector_space[0].grid]
    src_prime = g.eval(mat * src)
    dst.checkerboard(src_prime.checkerboard())
    X = g.inner_product(dst, src_prime)
    eps_ref = src.grid.precision.eps * finger_print_tolerance
    if mat.adj_mat is not None:
        X_from_adj = g.inner_product(src, g.adj(mat) * dst).conjugate()
        eps = abs(X - X_from_adj) / abs(X)
        g.message(f"Test adj({tag}): {eps}")
        assert eps < eps_ref
        if mat.inv_mat is not None:
            eps = (g.norm2(src - mat * g.inv(mat) * src) / g.norm2(src)) ** 0.5
            g.message(f"Test inv({tag}): {eps}")
            assert eps < eps_ref
            Y = g.inner_product(dst, g.inv(g.adj(mat)) * src)
            Y_from_adj = g.inner_product(src, g.inv(mat) * dst).conjugate()
            eps = abs(Y - Y_from_adj) / abs(Y)
            g.message(f"Test adj(inv({tag})): {eps}")
            assert eps < eps_ref
    # do even/odd tests
    even_odd_operators = {"": (".Mooee", ".Meooe")}
    if tag in even_odd_operators:
        g.message(f"Test eo versions of {tag}")
        grid_rb = fermion.F_grid_eo
        src_p = g.vspincolor(grid_rb)
        dst_p = g.vspincolor(grid_rb)

        tag_Mooee, tag_Meooe = even_odd_operators[tag]
        mat_Mooee = get_matrix(fermion, tag_Mooee)
        mat_Meooe = get_matrix(fermion, tag_Meooe)
        for parity in [g.even, g.odd]:
            g.pick_checkerboard(parity, src_p, src)
            g.pick_checkerboard(parity, dst_p, src)
            verify_matrix_element(fermion, dst_p, src_p, tag_Mooee)
            verify_projected_even_odd(mat, mat_Mooee, dst_p, src_p, src)

            g.pick_checkerboard(parity.inv(), dst_p, src)
            verify_matrix_element(fermion, dst_p, src_p, tag_Meooe)
            verify_projected_even_odd(mat, mat_Meooe, dst_p, src_p, src)
    # perform derivative tests
    projected_gradient_operators = {
        "": "M_projected_gradient",
        ".Dhop": "Dhop_projected_gradient",
        ".ImportPhysicalFermionSource": "ImportPhysicalFermionSource_projected_gradient",
    }
    if tag in projected_gradient_operators and isinstance(
        fermion, g.qcd.fermion.differentiable_fine_operator
    ):
        # Test projected gradient for src^dag M^dag M src
        g.message(f"Test projected_gradient of {tag} via src^dag M^dag M src")
        mat_pg = get_matrix(fermion, projected_gradient_operators[tag])
        dst_pg = g(mat * src)

        class df(g.group.differentiable_functional):
            def __call__(self, Uprime):
                return g.norm2(get_matrix(fermion.updated(Uprime), tag) * src)

            def gradient(self, Uprime, dUprime):
                assert dUprime == Uprime
                return [
                    g.qcd.gauge.project.traceless_hermitian(g.eval(a + b))
                    for a, b in zip(mat_pg(dst_pg, src), mat_pg.adj()(src, dst_pg))
                ]

        dfv = df()
        dfv.assert_gradient_error(rng, U, U, 1e-3, 1e-6)

        # Test projected gradient for src^dag G5 M src
        if isinstance(fermion, g.qcd.fermion.gauge_independent_g5_hermitian):
            g.message(f"Test projected_gradient of {tag} via src^dag G5 M src")

            class df(g.group.differentiable_functional):
                def __call__(self, Uprime):
                    return g.inner_product(
                        src, fermion.G5 * get_matrix(fermion.updated(Uprime), tag) * src
                    ).real

                def gradient(self, Uprime, dUprime):
                    assert dUprime == Uprime
                    return g.qcd.gauge.project.traceless_hermitian(mat_pg(fermion.G5 * src, src))

            dfv = df()
            dfv.assert_gradient_error(rng, U, U, 1e-3, 1e-6)

            g.message(f"Test projected_gradient of {tag} via src^dag M^dag G5 src")

            class df(g.group.differentiable_functional):
                def __call__(self, Uprime):
                    return g.inner_product(
                        src,
                        get_matrix(fermion.updated(Uprime), tag).adj() * fermion.G5 * src,
                    ).real

                def gradient(self, Uprime, dUprime):
                    assert dUprime == Uprime
                    return g.qcd.gauge.project.traceless_hermitian(
                        mat_pg.adj()(src, fermion.G5 * src)
                    )

            dfv = df()
            dfv.assert_gradient_error(rng, U, U, 1e-3, 1e-6)

    # perform even-odd derivative tests
    projected_gradient_operators = {".Meooe": "Meooe_projected_gradient"}
    if tag in projected_gradient_operators and isinstance(
        fermion, g.qcd.fermion.differentiable_fine_operator
    ):
        # Test projected gradient for src_p^dag M^dag M src_p
        g.message(f"Test projected_gradient of {tag} via src^dag M^dag M src")
        mat_pg = get_matrix(fermion, projected_gradient_operators[tag])
        src_p = g.lattice(fermion.F_grid_eo, fermion.otype)

        for parity in [g.even, g.odd]:
            g.pick_checkerboard(parity, src_p, src)
            dst_p = g(mat * src_p)

            class df(g.group.differentiable_functional):
                def __call__(self, Uprime):
                    return g.norm2(get_matrix(fermion.updated(Uprime), tag) * src_p)

                def gradient(self, Uprime, dUprime):
                    assert dUprime == Uprime
                    R = g.group.cartesian(Uprime)
                    for r, x in zip(R + R, mat_pg(dst_p, src_p) + mat_pg.adj()(src_p, dst_p)):
                        g.set_checkerboard(r, g.qcd.gauge.project.traceless_hermitian(x))
                    return R

        dfv = df()
        dfv.assert_gradient_error(rng, U, U, 1e-3, 1e-6)

    return X


g.default.set_verbose("random", False)

# test suite
for name in test_suite:
    # load configuration
    rng = g.random("finger_print")
    U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], g.double), rng)

    # default grid
    grid = U[0].grid

    # check tolerance
    eps = grid.precision.eps

    # params
    test = test_suite[name]
    g.message(f"\n\nStarting test suite for {name}")

    # create fermion
    fermion_dp = test["fermion"](U, test["params"])

    # make sure there are no reference loops in implementation
    if isinstance(fermion_dp, g.qcd.fermion.operator.base.base):
        g.message("Test for reference loops")
        interface = fermion_dp.interface
        del fermion_dp
        assert sys.getrefcount(interface) == 2
        del interface
        fermion_dp = test["fermion"](U, test["params"])

    # do full tests
    src = {}
    dst = {}
    for grid in [fermion_dp.F_grid, fermion_dp.U_grid]:
        if grid not in src:  # if F_grid != U_grid
            src[grid] = rng.cnormal(g.vspincolor(grid))
            dst[grid] = rng.cnormal(g.vspincolor(grid))

    # apply open boundaries to fields if necessary
    if test["params"]["boundary_phases"][-1] == 0.0:
        for grid in src:
            g.qcd.fermion.apply_open_boundaries(src[grid])
            g.qcd.fermion.apply_open_boundaries(dst[grid])

    for matrix in test["matrices"]:
        g.message(
            f"""

            Testing {name}{matrix}

"""
        )
        finger_print = []
        finger_print.append(verify_matrix_element(fermion_dp, dst, src, matrix))
        if test["matrices"][matrix] is None:
            g.message(f"Matrix {matrix} fingerprint: {finger_print}")
        else:
            fp = np.array(finger_print)
            eps = np.linalg.norm(fp - np.array(test["matrices"][matrix])) / np.linalg.norm(fp)
            g.message(f"Test {matrix} fingerprint: {eps}")
            assert eps < grid.precision.eps * finger_print_tolerance

    # test single versus double precision
    if isinstance(fermion_dp, g.qcd.fermion.fine_operator):
        fermion_sp = fermion_dp.converted(g.single)
        verify_single_versus_double_precision(rng, fermion_dp, fermion_sp)

    # test daggering entire fermion operator
    verify_daggered(rng, fermion_dp, fermion_dp.adj())
