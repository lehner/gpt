#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2025
#
import gpt as g
import numpy as np
import sys

# command line parameters
grid_f_size = g.default.get_ivec("--fgrid", [16, 16, 32, 16], 4)
grid_c_size = g.default.get_ivec("--cgrid", [8, 8, 8, 8], 4)
grid_cc_size = g.default.get_ivec("--ccgrid", [2, 2, 4, 8], 4)

# setup rng, make it quiet
g.default.set_verbose("random", False)
rng = g.random("test")

# setup fine link fields
U = g.qcd.gauge.random(g.grid(grid_f_size, g.double), rng)

# do everything in single precision
# U = g.convert(U, g.single)
eps2_threshold = 1e-20

# setup grids
grid_f = U[0].grid
grid_eo_f = grid_f.checkerboarded(g.redblack)
grid_c = g.grid(grid_c_size, grid_f.precision)
grid_cc = g.grid(grid_cc_size, grid_f.precision)

# number of basis vectors
nbasis_f = 20
nbasis_c = 12

# number of block orthogonalization steps
nblockortho = 1

# setup fine basis
basis_f = [g.vspincolor(grid_f) for __ in range(nbasis_f)]
rng.cnormal(basis_f)

basis_eo_f = [g.vspincolor(grid_eo_f) for __ in range(nbasis_f)]
rng.cnormal(basis_eo_f)

# setup coarse basis
basis_c = [g.vcomplex(grid_c, nbasis_f) for __ in range(nbasis_c)]
rng.cnormal(basis_c)

# setup fine block map
bm_f = g.block.map(grid_c, basis_f)
bm_eo_f = g.block.map(grid_c, basis_eo_f)
bm_c = g.block.map(grid_cc, basis_c)

# orthonormalize fine basis
for i in range(nblockortho):
    g.message("Block ortho step %d" % i)
    bm_f.orthonormalize()
    bm_eo_f.orthonormalize()
    bm_c.orthonormalize()


# tester
def test_coarse(slow_coarse, lpoints, tag, skip_eo_tests=False):
    g.message(
        f"================================================================================\nRun tests for {tag}"
    )
    cgrid = slow_coarse.vector_space[0].grid
    nbasis = len(slow_coarse.map.basis)

    # first find footprint
    if False:
        csrc = g.vcomplex(cgrid, nbasis)
        csrc[:] = 0
        csrc[0, 0, 0, 0, 0] = 1
        dd = g(slow_coarse * csrc)
        vv = dd[:, :, :, :]
        cc = g.coordinates(dd)
        vv = np.linalg.norm(vv, axis=1)
        vv = cc[vv > 1e-10]
        g.message(len(vv), "points in stencil")
        g.message(vv, lpoints)

    # sources for testing
    csrc = [rng.cnormal(g.vcomplex(cgrid, nbasis)) for i in range(6)]

    # compile fast version
    cop_st = slow_coarse.compile(lpoints)

    # compile fast version that has right-hand sides in grid-dimension 0
    # the .packed() makes it such that it can act on lists of right-hand side
    # the idea is to have the .packed() act on larger operations in packed state
    cop_packed = slow_coarse.compile(lpoints, packed_right_hand_sides=len(csrc)).packed()

    # warmup
    test2 = g(slow_coarse * csrc)
    test3 = g(cop_st * csrc)

    # check operator correctness
    tt = g.timer("apply")
    tt("projected fine")
    test2 = g(slow_coarse * csrc)
    tt("stencil")
    test3 = g(cop_st * csrc)
    tt("packed")
    test4 = g(cop_packed * csrc)
    tt()
    g.message(tt)
    for i in range(len(csrc)):
        eps2 = g.norm2(test3[i] - test2[i]) / g.norm2(test2[i])
        g.message(f"Test coarse operator against vector {i}: {eps2}")
        assert eps2 < eps2_threshold

        eps2 = g.norm2(test4[i] - test2[i]) / g.norm2(test2[i])
        g.message(f"Test coarse operator packing {i}: {eps2}")
        assert eps2 < eps2_threshold

    # checkerboarding?
    if cgrid.cb.n == 1 and not skip_eo_tests:
        cop_st_pc = cop_st.even_odd_sites_decomposed(g.even)
        cop_st_ee = cop_st_pc.DD
        cop_st_oo = cop_st_pc.CC
        cop_st_eo = cop_st_pc.DC
        cop_st_oe = cop_st_pc.CD

        # check even/odd
        g.message("check even/odd")

        csrc_e = []
        csrc_o = []
        dst_ee = []
        dst_oe = []
        dst_oo = []
        dst_eo = []
        for i in range(len(csrc)):
            csrc_e.append(g.pick_checkerboard(g.even, csrc[i]))
            csrc_o.append(g.pick_checkerboard(g.odd, csrc[i]))

            src_embedded_e = g(0 * csrc[i])
            g.set_checkerboard(src_embedded_e, csrc_e[i])
            dst_embedded_e = g(slow_coarse * src_embedded_e)

            dst_ee.append(g.pick_checkerboard(g.even, dst_embedded_e))
            dst_oe.append(g.pick_checkerboard(g.odd, dst_embedded_e))

            src_embedded_o = g(0 * csrc[i])
            g.set_checkerboard(src_embedded_o, csrc_o[i])
            dst_embedded_o = g(slow_coarse * src_embedded_o)

            dst_eo.append(g.pick_checkerboard(g.even, dst_embedded_o))
            dst_oo.append(g.pick_checkerboard(g.odd, dst_embedded_o))

        stencil_dst_ee = g(cop_st_ee * csrc_e)
        stencil_dst_oo = g(cop_st_oo * csrc_o)
        stencil_dst_eo = g(cop_st_eo * csrc_o)
        stencil_dst_oe = g(cop_st_oe * csrc_e)

        lhs = rng.cnormal(g.copy(csrc[0]))
        lhs_e = g.pick_checkerboard(g.even, lhs)
        lhs_o = g.pick_checkerboard(g.odd, lhs)

        # adj-ee
        A = g.inner_product(lhs_e, cop_st_ee * csrc_e[0]).conjugate()
        B = g.inner_product(csrc_e[0], cop_st_ee.adj() * lhs_e)
        assert abs((A - B) / A) < eps2_threshold**0.5

        # adj-oo
        A = g.inner_product(lhs_o, cop_st_oo * csrc_o[0]).conjugate()
        B = g.inner_product(csrc_o[0], cop_st_oo.adj() * lhs_o)
        assert abs((A - B) / A) < eps2_threshold**0.5

        # adj-eo
        A = g.inner_product(lhs_e, cop_st_eo * csrc_o[0]).conjugate()
        B = g.inner_product(csrc_o[0], cop_st_eo.adj() * lhs_e)
        assert abs((A - B) / A) < eps2_threshold**0.5

        if cop_st_ee.inv_mat is not None:
            g.message("Can test inverse")

            # inv-ee
            A = g(cop_st_ee.inv() * cop_st_ee * csrc_e[0])
            eps2 = g.norm2(A - csrc_e[0]) / g.norm2(A)
            assert eps2 < eps2_threshold

            # inv-oo
            A = g(cop_st_oo.inv() * cop_st_oo * csrc_o[0])
            eps2 = g.norm2(A - csrc_o[0]) / g.norm2(A)
            assert eps2 < eps2_threshold

            # adj-inv-ee
            A = g(cop_st_ee.inv().adj() * cop_st_ee.adj() * csrc_e[0])
            eps2 = g.norm2(A - csrc_e[0]) / g.norm2(A)
            assert eps2 < eps2_threshold

            # adj-inv-oo
            A = g(cop_st_oo.inv().adj() * cop_st_oo.adj() * csrc_o[0])
            eps2 = g.norm2(A - csrc_o[0]) / g.norm2(A)
            assert eps2 < eps2_threshold

        for i in range(len(csrc)):
            eps2 = g.norm2(dst_ee[i] - stencil_dst_ee[i]) / g.norm2(csrc_e[i])
            g.message("EE:", eps2)
            assert eps2 < eps2_threshold
            eps2 = g.norm2(dst_oo[i] - stencil_dst_oo[i]) / g.norm2(csrc_o[i])
            g.message("OO:", eps2)
            assert eps2 < eps2_threshold
            eps2 = g.norm2(dst_eo[i] - stencil_dst_eo[i]) / g.norm2(csrc_e[i])
            g.message("EO:", eps2)
            assert eps2 < eps2_threshold
            eps2 = g.norm2(dst_oe[i] - stencil_dst_oe[i]) / g.norm2(csrc_o[i])
            g.message("OE:", eps2)
            assert eps2 < eps2_threshold

    return cop_st


# test selection
# setup fine matrix
mat_f = g.qcd.fermion.wilson_clover(
    U,
    {
        "kappa": 0.137,
        "csw_r": 0,
        "csw_t": 0,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, -1.0],
    },
)

lpoints_9pt = [
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (-1, 0, 0, 0),
    (0, -1, 0, 0),
    (0, 0, -1, 0),
    (0, 0, 0, -1),
]

# force non-dir-disp stencil
lpoints_gen = lpoints_9pt + [
    (1, 0, 1, 0),
    (-1, 0, -1, 0),
]

coarse_1 = bm_f.coarse_operator(mat_f)


# now coarsen the the even-odd preconditioned operator
pc = g.qcd.fermion.preconditioner
Mpc = pc.eo1(parity=g.even)(mat_f).Mpc
lpoints_Mpc = [
    (x, y, z, t)
    for x in range(-1, 2)
    for y in range(-1, 2)
    for z in range(-1, 2)
    for t in range(-1, 2)
    if x**2 + y**2 + z**2 + t**2 <= 2
]
coarse_3 = bm_eo_f.coarse_operator(Mpc)


def _Mpc(dst, src):
    for i in range(len(dst)):
        dst[i] @= src[i] * 1.1 + 2.3j * g.cshift(src[i], 0, 2)
        dst[i].checkerboard(src[i].checkerboard())


Mpc_cart = g.matrix_operator(
    _Mpc,
    vector_space=g.vector_space.explicit_grid_otype_checkerboard(
        grid_eo_f, g.ot_vector_spin_color(4, 3), g.even
    ),
    accept_list=True,
)
lpoints_Mpc_cart = [(0, 0, 0, 0), (1, 0, 0, 0)]

coarse_3_cart = bm_eo_f.coarse_operator(Mpc_cart)

test_coarse(coarse_3_cart, lpoints_Mpc_cart, "e/o preconditooned fine -> coarse")
test_coarse(
    coarse_3,
    lpoints_Mpc,
    f"e/o preconditooned fine -> coarse (padded version) {len(lpoints_Mpc)}-point stencil",
    skip_eo_tests=True,  # with sufficient simd, padding loses too many factors of 2
)

fast_coarse_1 = test_coarse(coarse_1, lpoints_9pt, "9pt Wilson fine -> coarse")
# test_coarse(coarse_1, lpoints_gen, "9pt Wilson fine -> coarse (padded version)")

coarse_2 = bm_c.coarse_operator(fast_coarse_1)
test_coarse(coarse_2, lpoints_9pt, "9pt Wilson coarse -> coarse2")
# test_coarse(coarse_2, lpoints_gen, "9pt Wilson coarse -> coarse2 (padded version)")

# now test even-odd preconditioned inverse of coarse operator
inv = g.algorithms.inverter

g.default.set_verbose("block_cg_convergence")
g.default.set_verbose("block_cg_performance")
solver_pc = inv.preconditioned(
    # g.qcd.fermion.preconditioner.eo2_ne(parity=g.even),
    # g.qcd.fermion.preconditioner.eo2_ne(parity=g.odd),
    g.qcd.fermion.preconditioner.eo1_ne(parity=g.even),
    inv.block_cg({"eps": 1e-15, "maxiter": 1000}),
)(fast_coarse_1)

g.default.set_verbose("fgmres_convergence")
solver = inv.fgmres({"eps": 1e-15, "maxiter": 1000, "restartlen": 20})(fast_coarse_1)

test = [fast_coarse_1.vector_space[0].lattice() for i in range(6)]
rng.cnormal(test)

g.message("Coarse solve without and with preconditioning")
t = g.timer("solves")
t("fgmres")
res1 = g(solver * test)
t("pc_block_cg")
res2 = g(solver_pc * test)
t()
g.message(t)

for i in range(len(test)):
    eps2 = g.norm2(res1[i] - res2[i]) / g.norm2(res1[i])
    g.message(f"Solutions agree to: {eps2}")
    assert eps2 < eps2_threshold

# TODO:
# - need to re-pack list of vectors putting simd in nvector dimension (packing infrastructure, operator.packed, g.pack, g.unpack)
# - after packing infrastructure is there can efficiently select points to sum over ; add to matrix_vector local stencil the restriction to interior points in sum
# - add blas version of local_stencil.matrix_vector
