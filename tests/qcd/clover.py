#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#          Daniel Richtmann 2020
#
# Desc.: Compare faster clover term application with default one
#
import gpt as g

# setup rng, mute
g.default.set_verbose("random", False)
rng = g.random("clover")

for precision in [g.single, g.double]:
    # load configuration
    U = g.qcd.gauge.random(g.grid([8, 8, 8, 8], precision), rng)

    # default grid
    grid = U[0].grid

    # check tolerance
    eps = grid.precision.eps

    # fermion parameters
    wc_params = {
        "kappa": 0.13500,
        "csw_r": 1.978,
        "csw_t": 1.978,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, -1.0],
    }

    # fermion operators
    wc_default = g.qcd.fermion.wilson_clover(U, {**wc_params, **{"faster_apply": 0}})
    wc_new = g.qcd.fermion.wilson_clover(U, {**wc_params, **{"faster_apply": 1}})

    # fields
    src, dst_grid, dst_gpt = (
        g.vspincolor(grid),
        g.vspincolor(grid),
        g.vspincolor(grid),
    )
    src_e, src_o, dst_grid_e, dst_grid_o, dst_gpt_e, dst_gpt_o = (
        g.vspincolor(wc_default.F_grid_eo),
        g.vspincolor(wc_default.F_grid_eo),
        g.vspincolor(wc_default.F_grid_eo),
        g.vspincolor(wc_default.F_grid_eo),
        g.vspincolor(wc_default.F_grid_eo),
        g.vspincolor(wc_default.F_grid_eo),
    )
    rng.cnormal(src)
    g.pick_cb(g.even, src_e, src)
    g.pick_cb(g.odd, src_o, src)

    # test Mooee.adj_mat
    wc_default.Mooee.mat(dst_grid, src)
    wc_new.Mooee.mat(dst_gpt, src)
    wc_default.Mooee.mat(dst_grid_e, src_e)
    wc_new.Mooee.mat(dst_gpt_e, src_e)
    wc_default.Mooee.mat(dst_grid_o, src_o)
    wc_new.Mooee.mat(dst_gpt_o, src_o)
    rel_dev = g.norm2(dst_grid - dst_gpt) / g.norm2(dst_grid)
    rel_dev_e = g.norm2(dst_grid_e - dst_gpt_e) / g.norm2(dst_grid_e)
    rel_dev_o = g.norm2(dst_grid_o - dst_gpt_o) / g.norm2(dst_grid_o)
    g.message(
        f"""
    Test: Mooee.mat
    rel. dev. = {rel_dev} -> test {'passed' if rel_dev <= eps else 'failed'}
    rel. dev. even = {rel_dev_e} -> test {'passed' if rel_dev_e <= eps else 'failed'}
    rel. dev. odd = {rel_dev_o} -> test {'passed' if rel_dev_o <= eps else 'failed'}
        """
    )
    assert rel_dev <= eps
    assert rel_dev_e <= eps
    assert rel_dev_o <= eps

    # test Mooee.adj_mat
    wc_default.Mooee.adj_mat(dst_grid, src)
    wc_new.Mooee.adj_mat(dst_gpt, src)
    wc_default.Mooee.adj_mat(dst_grid_e, src_e)
    wc_new.Mooee.adj_mat(dst_gpt_e, src_e)
    wc_default.Mooee.adj_mat(dst_grid_o, src_o)
    wc_new.Mooee.adj_mat(dst_gpt_o, src_o)
    rel_dev = g.norm2(dst_grid - dst_gpt) / g.norm2(dst_grid)
    rel_dev_e = g.norm2(dst_grid_e - dst_gpt_e) / g.norm2(dst_grid_e)
    rel_dev_o = g.norm2(dst_grid_o - dst_gpt_o) / g.norm2(dst_grid_o)
    g.message(
        f"""
    Test: Mooee.adj_mat
    rel. dev. = {rel_dev} -> test {'passed' if rel_dev <= eps else 'failed'}
    rel. dev. even = {rel_dev_e} -> test {'passed' if rel_dev_e <= eps else 'failed'}
    rel. dev. odd = {rel_dev_o} -> test {'passed' if rel_dev_o <= eps else 'failed'}
        """
    )
    assert rel_dev <= eps
    assert rel_dev_e <= eps
    assert rel_dev_o <= eps

    # test Mooee.inv_mat
    wc_default.Mooee.inv_mat(dst_grid, src)
    wc_new.Mooee.inv_mat(dst_gpt, src)
    wc_default.Mooee.inv_mat(dst_grid_e, src_e)
    wc_new.Mooee.inv_mat(dst_gpt_e, src_e)
    wc_default.Mooee.inv_mat(dst_grid_o, src_o)
    wc_new.Mooee.inv_mat(dst_gpt_o, src_o)
    rel_dev = g.norm2(dst_grid - dst_gpt) / g.norm2(dst_grid)
    rel_dev_e = g.norm2(dst_grid_e - dst_gpt_e) / g.norm2(dst_grid_e)
    rel_dev_o = g.norm2(dst_grid_o - dst_gpt_o) / g.norm2(dst_grid_o)
    g.message(
        f"""
    Test: Mooee.inv_mat
    rel. dev. = {rel_dev} -> test {'passed' if rel_dev <= eps else 'failed'}
    rel. dev. even = {rel_dev_e} -> test {'passed' if rel_dev_e <= eps else 'failed'}
    rel. dev. odd = {rel_dev_o} -> test {'passed' if rel_dev_o <= eps else 'failed'}
        """
    )
    assert rel_dev <= eps
    assert rel_dev_e <= eps
    assert rel_dev_o <= eps

    # test Mooee.adj_inv_mat
    wc_default.Mooee.adj_inv_mat(dst_grid, src)
    wc_new.Mooee.adj_inv_mat(dst_gpt, src)
    wc_default.Mooee.adj_inv_mat(dst_grid_e, src_e)
    wc_new.Mooee.adj_inv_mat(dst_gpt_e, src_e)
    wc_default.Mooee.adj_inv_mat(dst_grid_o, src_o)
    wc_new.Mooee.adj_inv_mat(dst_gpt_o, src_o)
    rel_dev = g.norm2(dst_grid - dst_gpt) / g.norm2(dst_grid)
    rel_dev_e = g.norm2(dst_grid_e - dst_gpt_e) / g.norm2(dst_grid_e)
    rel_dev_o = g.norm2(dst_grid_o - dst_gpt_o) / g.norm2(dst_grid_o)
    g.message(
        f"""
    Test: Mooee.adj_inv_mat
    rel. dev. = {rel_dev} -> test {'passed' if rel_dev <= eps else 'failed'}
    rel. dev. even = {rel_dev_e} -> test {'passed' if rel_dev_e <= eps else 'failed'}
    rel. dev. odd = {rel_dev_o} -> test {'passed' if rel_dev_o <= eps else 'failed'}
        """
    )
    assert rel_dev <= eps
    assert rel_dev_e <= eps
    assert rel_dev_o <= eps
