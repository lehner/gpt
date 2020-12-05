#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Lorenzo Barca    (lorenzo1.barca@ur.de)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import gpt as g
import numpy as np
from gpt.qcd.spin_matrices import spin_matrix as spm
from gpt.qcd.baryon_contractions import BaryonsContractions2prop
from gpt.qcd.baryon_contractions import BaryonsContractions3prop
from gpt.qcd.heavy_baryon_contractions import HeavyBaryonsContractions2prop
from gpt.qcd.heavy_baryon_contractions import HeavyBaryonsContractions3prop
from gpt.qcd.create_hdf5 import _check_meas, _write_hdf5dset

b2p = BaryonsContractions2prop()
b3p = BaryonsContractions3prop()
hb2p = HeavyBaryonsContractions2prop()
hb3p = HeavyBaryonsContractions3prop()


def _heavy_baryon_loop(corr, quark_prop_1, quark_prop_2, quark_prop_3, moms, mom_list, time_rev, n_baryons):

    barid = 0

    meas = "Lambda_c-octetSU4(q3, q1, q1)_op18_lsc[1-2]"
    g.message(meas)
    tmp_correlator = hb3p._lambda8_2pt(quark_prop_3, quark_prop_1, quark_prop_1, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1


    meas = "Lambda_c-octetSU4(q1, q2, q3)_op18_lsc[1-2]"
    g.message(meas)
    tmp_correlator = hb3p._lambda8_2pt(quark_prop_3, quark_prop_1, quark_prop_2, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1


    meas = "Lambda_c-octetSU4_v2_op18_lsc[1-2]"
    g.message(meas)
    tmp_correlator = hb3p._lambdac_2pt_v2(quark_prop_3, quark_prop_1, quark_prop_1, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1


    meas = "Lambda_c-octetSU4_v2_op18_lsc[1-2]"
    g.message(meas)
    tmp_correlator = hb3p._lambdac_2pt_v2(quark_prop_1, quark_prop_2, quark_prop_3, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1


    meas = "Omega_cc^stSU4_op10_c[1-2]s"
    g.message(meas)
    tmp_correlator = b2p._sigma_star_2pt(quark_prop_2, quark_prop_1, spm.T_mixed(), spm.Cgm())
    g.message(type(tmp_correlator))

    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1


    meas = "Omega_ccSU4_op0_c[1-2]s"
    g.message(meas)
    tmp_correlator = hb2p._sigmac_GT_2pt(quark_prop_1, quark_prop_2, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    meas = "Omega_c^stSU4_op10_sc[1-2]"
    g.message(meas)
    tmp_correlator = b2p._sigma_star_2pt(quark_prop_2, quark_prop_1, spm.T_mixed(), spm.Cgm())

    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    meas = "Omega_cSU4_op0_sc[1-2]"
    g.message(meas)
    tmp_correlator = hb2p._sigmac_GT_2pt(quark_prop_1, quark_prop_2, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1


    meas = "Sigma_c^starSu4_op10_lc[1-2]"
    g.message(meas)
    tmp_correlator = b2p._sigma_star_2pt(quark_prop_2, quark_prop_1, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    meas = "Sigma_cSU4_op0_lc[1-2]"
    g.message(meas)
    tmp_correlator = hb2p._sigmac_GT_2pt(quark_prop_1, quark_prop_2, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1


    meas = "Xi_cc^starSu4_op10_c[1-2]l"
    g.message(meas)
    tmp_correlator = b2p._sigma_star_2pt(quark_prop_2, quark_prop_1, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
         g.message(mom_list[p_n])
         P = g.exp_ixp(p)
         correlator = g.slice(P * tmp_correlator, 3)
         for t, c in enumerate(correlator):
             g.message(t, c)
             corr[n_baryons * time_rev + barid, p_n, t] = c


    meas = "Xi_ccSU4_op0_c[1-2]l"
    g.message(meas)
    tmp_correlator = hb2p._sigmac_GT_2pt(quark_prop_1, quark_prop_2, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1


    meas = "Xi_c^prSU4_op0_lsc[1-2]"
    g.message(meas)
    tmp_correlator = hb3p._sigmac0_GT_2pt(quark_prop_1, quark_prop_2, quark_prop_3, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1


    meas = "Xi_c - Xi_c' mixing"
    g.message(meas)
    tmp_correlator = b3p._lambda8_to_sigma0_2pt(quark_prop_3, quark_prop_1, quark_prop_2, spm.T_unpol(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    meas = "Xi_c' - Xi_c mixing"
    g.message(meas)
    tmp_correlator = b3p._sigma0_to_lambda8_2pt(quark_prop_3, quark_prop_1, quark_prop_2, spm.T_unpol(), spm.Cg5())

    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    meas = "Xi_cSU4_op30_lsc[1-2]"
    g.message(meas)
    tmp_correlator = hb3p._lambda8_2pt(quark_prop_3, quark_prop_1, quark_prop_1, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1


    meas = "Omega_{ccc}"
    g.message(meas)
    tmp_correlator = b2p._sigma_star_2pt(quark_prop_3, quark_prop_3, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1


def heavy_baryon_spectrum(data_file, quark_prop_1, quark_prop_2, quark_prop_3, moms, mom_list, params):

    suN = params["su(n)"]
#    quarks_list = params["quarks"]
    kappa_list = params["kappa"]
    grid = quark_prop_1.grid
    Nt = grid.fdimensions[-1]
    n_baryons = 16

    correlators = np.zeros((n_baryons * 2, len(mom_list), Nt), dtype = complex)
    time_rev = 0
    _heavy_baryon_loop(correlators, quark_prop_1, quark_prop_2, quark_prop_3, moms, mom_list, time_rev, n_baryons)

    #
    # Time reversed propagators
    #
    print("Doing the time reversed measurements")
    quark_prop_1 = g.eval(-g.gamma[5] * g.gamma["T"] * quark_prop_1 * g.gamma[5] * g.gamma["T"])
    quark_prop_2 = g.eval(-g.gamma[5] * g.gamma["T"] * quark_prop_2 * g.gamma[5] * g.gamma["T"])
    quark_prop_3 = g.eval(-g.gamma[5] * g.gamma["T"] * quark_prop_3 * g.gamma[5] * g.gamma["T"])
    time_rev = 1

    _heavy_baryon_loop(correlators, quark_prop_1, quark_prop_2, quark_prop_3, moms, mom_list, time_rev, n_baryons)
    _write_hdf5dset(correlators, data_file)
