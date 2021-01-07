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
from gpt.qcd.create_hdf5 import _check_meas, _write_hdf5dset_baryon

b2p = BaryonsContractions2prop()
b3p = BaryonsContractions3prop()


def _light_baryon_loop(corr, light_quark_prop, strange_quark_prop, moms, mom_list, time_rev, n_baryons):

    barid = 0

    polarizations = {"polx":spm.T_polx(), "poly":spm.T_poly(), "polz":spm.T_polz(), \
                     "unpol":spm.T_unpol(), "mixed":spm.T_mixed()}

    for ii, pol in enumerate(["polx", "poly", "polz", "unpol"]):
        g.message(f"proton {pol}")
        polarization_matrix = polarizations[pol]
        tmp_correlator = b2p._proton_2pt(light_quark_prop, light_quark_prop, polarization_matrix, spm.Cg5())
        for p_n, p in enumerate(moms):
            g.message(mom_list[p_n])
            P = g.exp_ixp(p)
            correlator = g.slice(P * tmp_correlator, 3)
            for t, c in enumerate(correlator):
                g.message(t, c) 
                corr[n_baryons * time_rev + barid, p_n, t] = c
        barid += 1 # 0 - 3

    g.message("\Delta mixed")
    tmp_correlator = b2p._sigma_star_2pt(light_quark_prop, light_quark_prop, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
        corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1 # 4

    # in the isospin symmetry limit, the octet-\Sigma has the 
    # same diquark structure as the nucleon.
    #
    # TO CHECK SINCE I FIND DIFFERENT CONTRACTIONS
    # SIMILAR TO CHARMED SIGMA
    #
    for ii, pol in enumerate(["polx", "poly", "polz", "unpol"]):
        g.message(f"\Sigma-octet {pol}")
        polarization_matrix = polarizations[pol]
        tmp_correlator = b2p._proton_2pt(light_quark_prop, strange_quark_prop, polarization_matrix, spm.Cg5())
        for p_n, p in enumerate(moms):
            g.message(mom_list[p_n])
            P = g.exp_ixp(p)
            correlator = g.slice(P * tmp_correlator, 3)
            for t, c in enumerate(correlator):
                g.message(t, c)
                corr[n_baryons * time_rev + barid, p_n, t] = c
        barid += 1 # 5 - 8
    '''
    g.message("\Sigma_star-decuplet mixed")
    tmp_correlator = b2p._sigma_star_2pt(light_quark_prop, strange_quark_prop, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1 # 9
    '''

    g.message("\Sigma_star-decuplet mixed")
    tmp_correlator = b2p._sigma_star_2pt(strange_quark_prop, light_quark_prop, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1 # 9

    # \Xi-octet has similar Wick-contractions as the proton
    # \Xi-decuplet had similar Wick-contractions as the \Delta

    for ii, pol in enumerate(["polx", "poly", "polz", "unpol"]):
        g.message(f"\Xi-octet {pol}")
        polarization_matrix = polarizations[pol]
        tmp_correlator = b2p._proton_2pt(strange_quark_prop, light_quark_prop, polarization_matrix, spm.Cg5())
        for p_n, p in enumerate(moms):
            g.message(mom_list[p_n])
            P = g.exp_ixp(p)
            correlator = g.slice(P * tmp_correlator, 3)
            for t, c in enumerate(correlator):
                g.message(t, c)
                corr[n_baryons * time_rev + barid, p_n, t] = c
        barid += 1 # 10 - 13

    '''
    g.message("\Xi_star-decuplet mixed")
    tmp_correlator = b2p._sigma_star_2pt(strange_quark_prop, light_quark_prop, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1 # 14
    '''
    g.message("\Xi_star-decuplet mixed")
    tmp_correlator = b2p._sigma_star_2pt(light_quark_prop, strange_quark_prop, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1 # 14

    for ii, pol in enumerate(["polx", "poly", "polz", "unpol"]):
        g.message(f"\Lambda {pol}")
        polarization_matrix = polarizations[pol]
        tmp_correlator = b3p._lambda8_2pt(strange_quark_prop, light_quark_prop, light_quark_prop, polarization_matrix, spm.Cg5())
        #tmp_correlator = b3p._lambda8_2pt(light_quark_prop, light_quark_prop, strange_quark_prop, polarization_matrix, spm.Cg5())
        for p_n, p in enumerate(moms):
            g.message(mom_list[p_n])
            P = g.exp_ixp(p)
            correlator = g.slice(P * tmp_correlator, 3)
            for t, c in enumerate(correlator):
                g.message(t, c)
                corr[n_baryons * time_rev + barid, p_n, t] = c
        barid += 1 # 15 - 18

    for ii, pol in enumerate(["polx", "poly", "polz", "unpol"]):
        g.message(f"\Lambda-naive {pol}")
        polarization_matrix = polarizations[pol]
        tmp_correlator = b2p._lambda_naive_2pt(light_quark_prop, strange_quark_prop, polarization_matrix, spm.Cg5())
        for p_n, p in enumerate(moms):
            g.message(mom_list[p_n])
            P = g.exp_ixp(p)
            correlator = g.slice(P * tmp_correlator, 3)
            for t, c in enumerate(correlator):
                g.message(t, c)
                corr[n_baryons * time_rev + barid, p_n, t] = c
        barid += 1 # 19 - 22

    g.message("Omega_{sss}")
    tmp_correlator = b2p._sigma_star_2pt(strange_quark_prop, strange_quark_prop, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1 # 23


def light_baryon_spectrum(data_file, light_quark_prop, strange_quark_prop, moms, mom_list, params):

    tsrc_meas = _check_meas(data_file, "baryonspec")
    g.message(f"meas:{tsrc_meas}")
    kappa_list = params["kappa"]
    grid = light_quark_prop.grid
    Nt = grid.fdimensions[-1]
    n_baryons = 24

    correlators = np.zeros((n_baryons * 2, len(mom_list), Nt), dtype = complex)
    time_rev = 0
    _light_baryon_loop(correlators, light_quark_prop, strange_quark_prop, moms, mom_list, time_rev, n_baryons)

    g.mem_report()
    #
    # Time reversed propagators
    #
    print("Doing the time reversed measurements")
    light_quark_prop = g.eval(-g.gamma[5] * g.gamma["T"] * light_quark_prop * g.gamma[5] * g.gamma["T"])
    strange_quark_prop = g.eval(-g.gamma[5] * g.gamma["T"] * strange_quark_prop * g.gamma[5] * g.gamma["T"])
    time_rev = 1

    _light_baryon_loop(correlators, light_quark_prop, strange_quark_prop, moms, mom_list, time_rev, n_baryons)
    _write_hdf5dset_baryon(correlators, data_file, tsrc_meas)


