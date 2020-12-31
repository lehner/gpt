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

#
#  CHECK the correct insertion of flavors 
#

def _heavy_baryon_loop(corr, light_quark_prop, strange_quark_prop, heavy_quark_prop, moms, mom_list, time_rev, n_baryons):

    barid = 24

    polarizations = {"T_unpol":spm.T_unpol(), "T_mixed":spm.T_mixed()}

    # do L C first 
    g.message("charmed \Sigma unpol")
    tmp_correlator = hb2p._sigmac_GT_2pt(light_quark_prop, heavy_quark_prop, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
        corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    g.message("charmed \Sigma star mixed")
    tmp_correlator = b2p._sigma_star_2pt(heavy_quark_prop, light_quark_prop, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
        corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    # do C L  
    g.message("double charmed \Xi unpol")
    tmp_correlator = hb2p._sigmac_GT_2pt(heavy_quark_prop, light_quark_prop, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
        corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    g.message("double charmed \Xi star mixed")
    tmp_correlator = b2p._sigma_star_2pt(light_quark_prop, heavy_quark_prop, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
        corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    # do S C
    g.message("charmed \Omega unpol")
    tmp_correlator = hb2p._sigmac_GT_2pt(strange_quark_prop, heavy_quark_prop, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
        corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    g.message("charmed \Omega star mixed")
    tmp_correlator = b2p._sigma_star_2pt(heavy_quark_prop, strange_quark_prop, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
        corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    # do C S
    g.message("double charmed \Omega unpol")
    tmp_correlator = hb2p._sigmac_GT_2pt(heavy_quark_prop, strange_quark_prop, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
        corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    g.message("double charmed \Omega star mixed")
    tmp_correlator = b2p._sigma_star_2pt(strange_quark_prop, heavy_quark_prop, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
        corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    # do L S C
    g.message("charmed \Xi prime unpol")
    tmp_correlator = hb3p._sigmac0_GT_2pt(light_quark_prop, strange_quark_prop, charm_quark_prop, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
        corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1    

    # do L S C -> C L L (?)
    g.message("charmed \Lambda unpol")
    tmp_correlator = hb3p._lambda8_2pt(charm_quark_prop, light_quark_prop, light_quark_prop, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    #tmp_correlator = hb3p._lambda8_2pt(light_quark_prop, light_quark_prop, charm_quark_prop, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
        corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1
    
    # do L S C  -> C L S
    g.message("charmed \Xi unpol")
    tmp_correlator = hb3p._lambda8_2pt(charm_quark_prop, strange_quark_prop, light_quark_prop, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
        corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    g.message("Xi_c - Xi_c' mixing")
    tmp_correlator = b3p._lambda8_to_sigma0_2pt(quark_prop_3, quark_prop_1, quark_prop_2, spm.T_unpol(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    g.message("Xi_c' - Xi_c mixing")
    tmp_correlator = b3p._sigma0_to_lambda8_2pt(quark_prop_3, quark_prop_1, quark_prop_2, spm.T_unpol(), spm.Cg5())

    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    # do L S C -> C L L (?)
    g.message("charmed \Xi star mixed")
    tmp_correlator = b3p._sigma_star_2pt(light_quark_prop, charm_quark_prop, strange_quark_prop, spm.T_mixed(), spm.Cg5(), spm.Cg5())
    #tmp_correlator = hb3p._sigma_star_2pt(strange_quark_prop, light_quark_prop, charm_quark_prop, spm.T_unpol(), spm.Cg5(), spm.Cg5())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
        corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1

    g.message("Omega_{ccc}")
    tmp_correlator = b2p._sigma_star_2pt(charm_quark_prop, charm_quark_prop, spm.T_mixed(), spm.Cgm())
    for p_n, p in enumerate(moms):
        g.message(mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(P * tmp_correlator, 3)
        for t, c in enumerate(correlator):
            g.message(t, c)
            corr[n_baryons * time_rev + barid, p_n, t] = c
    barid += 1    


def heavy_baryon_spectrum(data_file, light_quark_prop, strange_quark_prop, charm_quark_prop, moms, mom_list, params):

    tsrc_meas = _check_meas(data_file, "baryonspec")    
    g.message(f"meas:{tsrc_meas}")
    kappa_list = params["kappa"]
    grid = light_quark_prop.grid
    Nt = grid.fdimensions[-1]
    n_baryons = 24

    correlators = np.zeros((n_baryons * 2, len(mom_list), Nt), dtype = complex)
    time_rev = 0
    _heavy_baryon_loop(correlators, light_quark_prop, strange_quark_prop, charm_quark_prop, moms, mom_list, time_rev, n_baryons)

    g.mem_report()
    #
    # Time reversed propagators
    #
    print("Doing the time reversed measurements")
    light_quark_prop = g.eval(-g.gamma[5] * g.gamma["T"] * light_quark_prop * g.gamma[5] * g.gamma["T"])
    strange_quark_prop = g.eval(-g.gamma[5] * g.gamma["T"] * strange_quark_prop * g.gamma[5] * g.gamma["T"])
    quark_prop_3 = g.eval(-g.gamma[5] * g.gamma["T"] * quark_prop_3 * g.gamma[5] * g.gamma["T"])
    time_rev = 1

    _light_baryon_loop(correlators, light_quark_prop, strange_quark_prop, charm_quark_prop, moms, mom_list, time_rev, n_baryons)
    _write_hdf5dset_baryon(correlators, data_file, tsrc_meas)


