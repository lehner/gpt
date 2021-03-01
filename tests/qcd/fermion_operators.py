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
    "xi_0": 1,
    "nu": 1,
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

# create point source
src = g.mspincolor(grid)
g.create.point(
    src, [1, 0, 0, 0]
)  # pick point 1 so that "S" in preconditioner contributes to test

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
q0prop = quark0.propagator(
    inv.preconditioned(pc.eo2_ne(), inv.cg(eps=1e-7, maxiter=1000))
)
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
rhq = g.qcd.fermion.rhq_columbia(
    U, mass=4.0, cp=3.0, zeta=2.5, boundary_phases=[1, 1, 1, -1]
)


#########################################################################
# Test fermion operators against known results
#
# this protects against bugs introduced in the matrix application
# of fermion operators (new architectures / implementations)
#
# These tests are only meaningfull if reference implementation is also
# included
#########################################################################
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
wilson_clover_matrices_rb = {
    ".Mooee": [
        (-964.604747199766 + 1955.3204837295887j),
        (-984.9379301006896 + 1812.470240028601j),
    ],
    ".Meooe": [
        (-180.76784134943463 - 907.7516272233529j),
        (563.2079135281185 - 720.151781978722j),
    ],
}
wilson_clover_matrices = {
    "": [(-946.8714968698364 - 427.1253034080037j)],
    ".Mdiag": [(-908.620454398646 - 3428.779878527792j)],
}
wilson_clover_matrices_rb_open = {
    ".Mooee": [
        (-786.8010863449176 + 1070.7776954461517j),
        (-799.9912213689927 + 868.0454954800668j),
    ],
    ".Meooe": [
        (-515.6416289919217 - 727.8655685760094j),
        (-186.16467422408803 - 815.6089918787717j),
    ],
}
wilson_clover_matrices_open = {
    "": [(-1634.2615676797234 + 239.27037187495998j)],
    ".Mdiag": [(-1239.3535155227526 - 1158.5295177146759j)],
}
test_suite = {
    "wilson_clover": {
        "fermion": g.qcd.fermion.wilson_clover,
        "params": wilson_clover_params,
        "matrices_rb": wilson_clover_matrices_rb,
        "matrices": wilson_clover_matrices,
    },
    "wilson_clover_legacy": {
        "fermion": g.qcd.fermion.wilson_clover,
        "params": {
            **wilson_clover_params,
            "use_legacy": True,
        },  # will be deprecated eventually
        "matrices_rb": wilson_clover_matrices_rb,
        "matrices": wilson_clover_matrices,
    },
    "wilson_clover_reference": {
        "fermion": g.qcd.fermion.reference.wilson_clover,
        "params": wilson_clover_params,
        "matrices_rb": wilson_clover_matrices_rb,
        "matrices": wilson_clover_matrices,
    },
    "wilson_clover_openbc_reference": {
        "fermion": g.qcd.fermion.reference.wilson_clover,
        "params": wilson_clover_open_params,
        "matrices_rb": wilson_clover_matrices_rb_open,
        "matrices": wilson_clover_matrices_open,
    },
    "wilson_clover_openbc": {
        "fermion": g.qcd.fermion.wilson_clover,
        "params": wilson_clover_open_params,
        "matrices_rb": wilson_clover_matrices_rb_open,
        "matrices": wilson_clover_matrices_open,
    },
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
        "matrices_rb": {
            ".Mooee": [
                (-2446.6009975599286 + 5633.208069699162j),
                (-2446.6009975599286 + 5633.208069699162j),
            ],
            ".Meooe": [
                (2320.1432377385536 - 2611.1244058385605j),
                (3083.8202044273708 - 1957.829867279577j),
            ],
        },
        "matrices": {
            "": [
                (-2424.048033434305 + 10557.661684178218j),
            ],
            ".Mdiag": [(2643.396577965267 + 6550.259431381319j)],
        },
    },
}


def verify_matrix_element(mat, dst, src, tag):
    src_prime = g.eval(mat * src)
    dst.checkerboard(src_prime.checkerboard())
    X = g.inner_product(dst, src_prime)
    eps_ref = src.grid.precision.eps * 50.0
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
    return X


g.default.set_verbose("random", False)
for precision in [g.single, g.double]:
    # test suite
    for name in test_suite:

        # load configuration
        rng = g.random("finger_print")
        U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], precision), rng)

        # default grid
        grid = U[0].grid

        # check tolerance
        eps = grid.precision.eps

        # params
        test = test_suite[name]
        g.message(f"Starting test suite for {precision.__name__} precision {name}")

        # create fermion
        fermion = test["fermion"](U, test["params"])

        # do red/black tests
        grid_rb = fermion.F_grid_eo
        src = rng.cnormal(g.vspincolor(grid_rb))
        dst = rng.cnormal(g.vspincolor(grid_rb))

        # apply open boundaries to fields if necessary
        if test["params"]["boundary_phases"][-1] == 0.0:
            g.qcd.fermion.apply_open_boundaries(src)
            g.qcd.fermion.apply_open_boundaries(dst)

        g.message(f"<dst|src> = {g.inner_product(dst, src)}")
        for matrix in test["matrices_rb"]:
            finger_print = []
            for cb in [g.even, g.odd]:
                src.checkerboard(cb)
                mat = eval(f"fermion{matrix}")
                finger_print.append(
                    verify_matrix_element(mat, dst, src, f"fermion{matrix}")
                )
            if test["matrices_rb"][matrix] is None:
                g.message(f"Matrix {matrix} fingerprint: {finger_print}")
            else:
                fp = np.array(finger_print)
                eps = np.linalg.norm(
                    fp - np.array(test["matrices_rb"][matrix])
                ) / np.linalg.norm(fp)
                g.message(f"Test {matrix} fingerprint: {eps}")
                if eps >= grid.precision.eps * 10.0:
                    g.message(finger_print)
                    g.message(test["matrices_rb"][matrix])
                assert eps < grid.precision.eps * 10.0

        # do full tests
        grid = fermion.F_grid
        src = rng.cnormal(g.vspincolor(grid))
        dst = rng.cnormal(g.vspincolor(grid))

        # apply open boundaries to fields if necessary
        if test["params"]["boundary_phases"][-1] == 0.0:
            g.qcd.fermion.apply_open_boundaries(src)
            g.qcd.fermion.apply_open_boundaries(dst)

        for matrix in test["matrices"]:
            finger_print = []
            mat = eval(f"fermion{matrix}")
            finger_print.append(
                verify_matrix_element(mat, dst, src, f"fermion{matrix}")
            )
            if test["matrices"][matrix] is None:
                g.message(f"Matrix {matrix} fingerprint: {finger_print}")
            else:
                fp = np.array(finger_print)
                eps = np.linalg.norm(
                    fp - np.array(test["matrices"][matrix])
                ) / np.linalg.norm(fp)
                g.message(f"Test {matrix} fingerprint: {eps}")
                assert eps < grid.precision.eps * 10.0
