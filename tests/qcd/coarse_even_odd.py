#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test even-odd preconditioning for coarse operator
#
import gpt as g
import numpy as np
import sys

# command line parameters
grid_f_size = g.default.get_ivec("--fgrid", [16, 16, 16, 16], 4)
grid_c_size = g.default.get_ivec("--cgrid", [8, 8, 8, 8], 4)

# setup rng, make it quiet
g.default.set_verbose("random", False)
rng = g.random("test")

# setup fine link fields
U = g.qcd.gauge.random(g.grid(grid_f_size, g.single), rng)

# setup grids
grid_f = U[0].grid
grid_c = g.grid(grid_c_size, grid_f.precision)

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
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)

# number of basis vectors
nbasis_f = 30

# number of block orthogonalization steps
nblockortho = 1

# setup fine basis
basis_f = [g.vspincolor(grid_f) for __ in range(nbasis_f // 2)]
rng.cnormal(basis_f)

# split fine basis into chiral halfs
g.qcd.fermion.coarse.split_chiral(basis_f)

# setup fine block map
bm_f = g.block.map(grid_c, basis_f)

# orthonormalize fine basis
for i in range(nblockortho):
    g.message("Block ortho step %d" % i)
    bm_f.orthonormalize()

# create coarse link fields
A_c = [g.mcomplex(grid_c, nbasis_f) for __ in range(9)]
g.qcd.fermion.coarse.create_links(
    A_c, mat_f, basis_f, {"make_hermitian": False, "save_links": True}
)

# create coarse operator from links
mat_c = g.qcd.fermion.coarse_fermion(A_c, level=0)


# save typing
def vec_c_full():
    return g.vcomplex(mat_c.F_grid, nbasis_f)


def vec_c_half():
    return g.vcomplex(mat_c.F_grid_eo, nbasis_f)


def vec_f_full():
    return g.vspincolor(mat_f.F_grid)


def vec_f_half():
    return g.vspincolor(mat_f.F_grid_eo)


# setup coarse vectors
vec_in_f, vec_out_f = vec_f_full(), vec_f_full()
vec_in_c, vec_out_c = vec_c_full(), vec_c_full()
vec_in_f_eo, vec_out_f_eo = vec_f_half(), vec_f_half()
vec_in_c_eo, vec_out_c_eo = vec_c_half(), vec_c_half()
rng.cnormal(vec_in_f)
rng.cnormal(vec_in_c)
rng.cnormal(vec_in_f_eo)
rng.cnormal(vec_in_c_eo)
vec_in_f_eo.checkerboard(g.odd)
vec_in_c_eo.checkerboard(g.odd)

# test coarse link inversion
# NOTE: can't multiply and test for identity directly so we act on vector
A_self = A_c[8]
A_self_inv = g.matrix.inv(A_self)
vec_out_c @= A_self_inv * A_self * vec_in_c
assert g.norm2(vec_out_c - vec_in_c) / g.norm2(vec_in_c) < 1e-13
g.message("Test: coarse link inversion: passed\n")

# test correct checkerboards on full grid
for m, o, i, w in [
    (mat_f, vec_out_f, vec_in_f, "fine"),
    (mat_c, vec_out_c, vec_in_c, "coarse"),
]:
    for op in [m.mat, m.adj_mat, m.Mdiag.mat, m.Mdir(0, 1).mat]:
        op(o, i)
        assert o.checkerboard() == i.checkerboard()
    g.message(f"Test: same checkerboard operations on {w} full grid: all passed\n")

# test correct checkerboards on half grid
for m, o, i, w in [
    (mat_f, vec_out_f_eo, vec_in_f_eo, "fine"),
    (mat_c, vec_out_c_eo, vec_in_c_eo, "coarse"),
]:
    for op in [m.Mooee.mat, m.Mooee.adj_mat, m.Mooee.inv_mat, m.Mooee.adj_inv_mat]:
        op(o, i)
        assert o.checkerboard() == i.checkerboard()
    g.message(f"Test: same checkerboard operations on {w} half grid: all passed\n")

    for op in [m.Meooe.mat, m.Meooe.adj_mat]:
        op(o, i)
        assert o.checkerboard() == i.checkerboard().inv()
    g.message(f"Test: different checkerboard operations on {w} half grid: all passed\n")

# test for correctness of eo routines
for mat, vec_full, vec_half, w in [
    (mat_f, vec_f_full, vec_f_half, "fine"),
    (mat_c, vec_c_full, vec_c_half, "coarse"),
]:
    # setup vectors
    src, tmp, res, ref = (
        vec_full(),
        vec_full(),
        vec_full(),
        vec_full(),
    )
    src_e, src_o, tmp_e, tmp_o, tmp2_e, tmp2_o, res_e, res_o = (
        vec_half(),
        vec_half(),
        vec_half(),
        vec_half(),
        vec_half(),
        vec_half(),
        vec_half(),
        vec_half(),
    )
    rng.cnormal(src)
    g.pick_checkerboard(g.even, src_e, src)
    g.pick_checkerboard(g.odd, src_o, src)

    # Meo + Moe = Dhop
    mat.Dhop.mat(ref, src)
    mat.Meooe.mat(res_o, src_e)
    mat.Meooe.mat(res_e, src_o)
    g.set_checkerboard(res, res_e)
    g.set_checkerboard(res, res_o)
    rel_dev = g.norm2(ref - res) / g.norm2(ref)
    g.message(
        f"""
Test: Meo + Moe = Dhop
    src = {g.norm2(src)}
    ref = {g.norm2(ref)}
    res = {g.norm2(res)}
    rel. dev. = {rel_dev} -> test {'passed' if rel_dev <= 1e-15 else 'failed'}"""
    )
    assert rel_dev <= 1e-15

    # Meo^dag + Moe^dag = Dhop^dag
    mat.Dhop.adj_mat(ref, src)
    mat.Meooe.adj_mat(res_o, src_e)
    mat.Meooe.adj_mat(res_e, src_o)
    g.set_checkerboard(res, res_e)
    g.set_checkerboard(res, res_o)
    rel_dev = g.norm2(ref - res) / g.norm2(ref)
    g.message(
        f"""
Test: Meo^dag + Moe^dag = Dhop^dag
    src = {g.norm2(src)}
    ref = {g.norm2(ref)}
    res = {g.norm2(res)}
    rel. dev. = {rel_dev} -> test {'passed' if rel_dev <= 1e-15 else 'failed'}"""
    )
    assert rel_dev <= 1e-15

    # Meo + Moe + Moo + Mee = M
    mat.mat(ref, src)
    mat.Mooee.mat(tmp_e, src_e)
    mat.Mooee.mat(tmp_o, src_o)
    mat.Meooe.mat(tmp2_o, src_e)
    mat.Meooe.mat(tmp2_e, src_o)
    tmp2_o += tmp_o
    tmp2_e += tmp_e
    g.set_checkerboard(res, tmp2_e)
    g.set_checkerboard(res, tmp2_o)
    rel_dev = g.norm2(ref - res) / g.norm2(ref)
    g.message(
        f"""
Test: Meo + Moe + Moo + Mee = M
    src = {g.norm2(src)}
    ref = {g.norm2(ref)}
    res = {g.norm2(res)}
    rel. dev. = {rel_dev} -> test {'passed' if rel_dev <= 1e-14 else 'failed'}"""
    )
    assert rel_dev <= 1e-13

    # Meo^dag + Moe^dag + Moo^dag + Mee^dag = M^dag
    mat.adj_mat(ref, src)
    mat.Mooee.adj_mat(tmp_e, src_e)
    mat.Mooee.adj_mat(tmp_o, src_o)
    mat.Meooe.adj_mat(tmp2_o, src_e)
    mat.Meooe.adj_mat(tmp2_e, src_o)
    tmp2_o += tmp_o
    tmp2_e += tmp_e
    g.set_checkerboard(res, tmp2_e)
    g.set_checkerboard(res, tmp2_o)
    rel_dev = g.norm2(ref - res) / g.norm2(ref)
    g.message(
        f"""
Test: Meo^dag + Moe^dag + Moo^dag + Mee^dag = M^dag
    src = {g.norm2(src)}
    ref = {g.norm2(ref)}
    res = {g.norm2(res)}
    rel. dev. = {rel_dev} -> test {'passed' if rel_dev <= 1e-14 else 'failed'}"""
    )
    assert rel_dev <= 1e-14

    # imag(v^dag M^dag M v) = 0 (on full grid)
    mat.mat(tmp, src)
    mat.adj_mat(res, tmp)
    dot = g.inner_product(src, res)
    rel_dev = abs(dot.imag) / abs(dot.real)
    g.message(
        f"""
Test: imag(v^dag M^dag M v) = 0 (on full grid)
    dot = {dot}
    rel. dev. = {rel_dev} -> test {'passed' if rel_dev <= 1e-9 else 'failed'}"""
    )
    assert rel_dev <= 1e-9

    # imag(v^dag Meooe^dag Meooe v) = 0 (on both cbs)
    mat.Meooe.mat(tmp_o, src_e)
    mat.Meooe.adj_mat(res_e, tmp_o)
    mat.Meooe.mat(tmp_e, src_o)
    mat.Meooe.adj_mat(res_o, tmp_e)
    dot_e = g.inner_product(src_e, res_e)
    dot_o = g.inner_product(src_o, res_o)
    rel_dev_e = abs(dot_e.imag) / abs(dot_e.real)
    rel_dev_o = abs(dot_o.imag) / abs(dot_o.real)
    g.message(
        f"""
Test: imag(v^dag Meooe^dag Meooe v) = 0 (on both cbs)
    dot_e = {dot_e}
    dot_o = {dot_o}
    rel. dev. even = {rel_dev_e} -> test {'passed' if rel_dev_e <= 1e-8 else 'failed'}
    rel. dev. odd = {rel_dev_o} -> test {'passed' if rel_dev_o <= 1e-8 else 'failed'}"""
    )
    assert rel_dev_e <= 1e-8 and rel_dev_o <= 1e-8

    # imag(v^dag Mooee^dag Mooee v) = 0 (on full grid + both cbs)
    mat.Mooee.mat(tmp, src)
    mat.Mooee.adj_mat(res, tmp)
    mat.Mooee.mat(tmp_e, src_e)
    mat.Mooee.adj_mat(res_e, tmp_e)
    mat.Mooee.mat(tmp_o, src_o)
    mat.Mooee.adj_mat(res_o, tmp_o)
    dot = g.inner_product(src, res)
    dot_e = g.inner_product(src_e, res_e)
    dot_o = g.inner_product(src_o, res_o)
    rel_dev = abs(dot.imag) / abs(dot.real)
    rel_dev_e = abs(dot_e.imag) / abs(dot_e.real)
    rel_dev_o = abs(dot_o.imag) / abs(dot_o.real)
    g.message(
        f"""
Test: imag(v^dag Mooee^dag Mooee v) = 0 (on full grid + both cbs)
    dot = {dot}
    dot_e = {dot_e}
    dot_o = {dot_o}
    rel. dev. = {rel_dev} -> test {'passed' if rel_dev <= 1e-8 else 'failed'}
    rel. dev. even = {rel_dev_e} -> test {'passed' if rel_dev_e <= 1e-8 else 'failed'}
    rel. dev. odd = {rel_dev_o} -> test {'passed' if rel_dev_o <= 1e-8 else 'failed'}"""
    )
    assert rel_dev <= 1e-8 and rel_dev_e <= 1e-8 and rel_dev_o <= 1e-8

    # imag(v^dag Mooee^-dag Mooee^-1 v) = 0 (on full grid + both cbs)
    mat.Mooee.inv_mat(tmp, src)
    mat.Mooee.adj_inv_mat(res, tmp)
    mat.Mooee.inv_mat(tmp_e, src_e)
    mat.Mooee.adj_inv_mat(res_e, tmp_e)
    mat.Mooee.inv_mat(tmp_o, src_o)
    mat.Mooee.adj_inv_mat(res_o, tmp_o)
    dot = g.inner_product(src, res)
    dot_e = g.inner_product(src_e, res_e)
    dot_o = g.inner_product(src_o, res_o)
    rel_dev = abs(dot.imag) / abs(dot.real)
    rel_dev_e = abs(dot_e.imag) / abs(dot_e.real)
    rel_dev_o = abs(dot_o.imag) / abs(dot_o.real)
    g.message(
        f"""
Test: imag(v^dag Mooee^-dag Mooee^-1 v) = 0 (on full grid + both cbs)
    dot = {dot}
    dot_e = {dot_e}
    dot_o = {dot_o}
    rel. dev. = {rel_dev} -> test {'passed' if rel_dev <= 1e-8 else 'failed'}
    rel. dev. even = {rel_dev_e} -> test {'passed' if rel_dev_e <= 1e-8 else 'failed'}
    rel. dev. odd = {rel_dev_o} -> test {'passed' if rel_dev_o <= 1e-8 else 'failed'}"""
    )
    assert rel_dev <= 1e-8 and rel_dev_e <= 1e-8 and rel_dev_o <= 1e-8

    # Mooee^-1 Mooee = 1 (on full grid + both cbs)
    mat.Mooee.mat(tmp, src)
    mat.Mooee.inv_mat(res, tmp)
    mat.Mooee.mat(tmp_e, src_e)
    mat.Mooee.inv_mat(res_e, tmp_e)
    mat.Mooee.mat(tmp_o, src_o)
    mat.Mooee.inv_mat(res_o, tmp_o)
    rel_dev = g.norm2(src - res) / g.norm2(src)
    rel_dev_e = g.norm2(src_e - res_e) / g.norm2(src_e)
    rel_dev_o = g.norm2(src_o - res_o) / g.norm2(src_o)
    g.message(
        f"""
Test: Mooee^-1 Mooee = 1 (on full grid + both cbs)
    src = {g.norm2(src)}
    src_e = {g.norm2(src_e)}
    src_o = {g.norm2(src_o)}
    res = {g.norm2(res)}
    res_e = {g.norm2(res_e)}
    res_o = {g.norm2(res_o)}
    rel. dev. = {rel_dev} -> test {'passed' if rel_dev <= 1e-13 else 'failed'}
    rel. dev. even = {rel_dev_e} -> test {'passed' if rel_dev_e <= 1e-13 else 'failed'}
    rel. dev. odd = {rel_dev_o} -> test {'passed' if rel_dev_o <= 1e-13 else 'failed'}"""
    )
    assert rel_dev <= 1e-13 and rel_dev_e <= 1e-13 and rel_dev_o <= 1e-13

    # Mooee^-dag Mooee^dag = 1 (on full grid + both cbs)
    mat.Mooee.adj_mat(tmp, src)
    mat.Mooee.adj_inv_mat(res, tmp)
    mat.Mooee.adj_mat(tmp_e, src_e)
    mat.Mooee.adj_inv_mat(res_e, tmp_e)
    mat.Mooee.adj_mat(tmp_o, src_o)
    mat.Mooee.adj_inv_mat(res_o, tmp_o)
    rel_dev = g.norm2(src - res) / g.norm2(src)
    rel_dev_e = g.norm2(src_e - res_e) / g.norm2(src_e)
    rel_dev_o = g.norm2(src_o - res_o) / g.norm2(src_o)
    g.message(
        f"""
Test: Mooee^-dag Mooee^dag = 1 (on full grid + both cbs)
    src = {g.norm2(src)}
    src_e = {g.norm2(src_e)}
    src_o = {g.norm2(src_o)}
    res = {g.norm2(res)}
    res_e = {g.norm2(res_e)}
    res_o = {g.norm2(res_o)}
    rel. dev. = {rel_dev} -> test {'passed' if rel_dev <= 1e-13 else 'failed'}
    rel. dev. even = {rel_dev_e} -> test {'passed' if rel_dev_e <= 1e-13 else 'failed'}
    rel. dev. odd = {rel_dev_o} -> test {'passed' if rel_dev_o <= 1e-13 else 'failed'}"""
    )
    assert rel_dev <= 1e-13 and rel_dev_e <= 1e-13 and rel_dev_o <= 1e-13

    # imag(v^dag Mpc v) = 0 (on both cbs)
    mat_pc = g.qcd.fermion.preconditioner.eo2_ne(parity=g.odd)(mat)
    mat_pc.Mpc(res_e, src_e)
    mat_pc.Mpc(res_o, src_o)
    dot_e = g.inner_product(src_e, res_e)
    dot_o = g.inner_product(src_o, res_o)
    rel_dev_e = abs(dot_e.imag) / abs(dot_e.real)
    rel_dev_o = abs(dot_o.imag) / abs(dot_o.real)
    g.message(
        f"""
Test: imag(v^dag Mpc v) = 0 (on both cbs)
    dot_e = {dot_e}
    dot_o = {dot_o}
    rel. dev. even = {rel_dev_e} -> test {'passed' if rel_dev_e <= 1e-8 else 'failed'}
    rel. dev. odd = {rel_dev_o} -> test {'passed' if rel_dev_o <= 1e-8 else 'failed'}"""
    )
    assert rel_dev_e <= 1e-8 and rel_dev_o <= 1e-8

    g.message(f"\nTest: operations on {w} grid: all passed\n")

# abbreviations
i = g.algorithms.inverter
p = g.qcd.fermion.preconditioner

# solver objects
slv_alg = i.fgmres({"eps": 1e-6, "maxiter": 100})
slv_full = slv_alg
slv_eo2 = i.preconditioned(p.eo2(parity=g.odd), slv_alg)

# solve on full grid
sol_full = g.eval(slv_full(mat_c) * vec_in_c)
eps2 = g.norm2(mat_c * sol_full - vec_in_c) / g.norm2(vec_in_c)
niter_full = len(slv_alg.history)
g.message(f"eps2 full: {eps2}")
assert eps2 < 1e-8

# solve on eo grid
sol_eo = g.eval(slv_eo2(mat_c) * vec_in_c)
eps2 = g.norm2(mat_c * sol_eo - vec_in_c) / g.norm2(vec_in_c)
niter_eo = len(slv_alg.history)
g.message(f"eps2 eo: {eps2}")
assert eps2 < 1e-8

# require that eo performs better
g.message(f"Test: niter_eo < niter_full: {'passed' if niter_eo < niter_full else 'failed'}")
assert niter_eo < niter_full
