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
from gpt.qcd.create_hdf5 import _check_meas, _write_hdf5dset

b2p = BaryonsContractions2prop()
b3p = BaryonsContractions2prop()


def _baryon_loop(corr, quark_prop_1, quark_prop_2, quark_prop_3, moms, mom_list, time_rev, n_baryons):

    for baryon_n in range(n_baryons):
        if(baryon_n == 0):
            meas = "Sigma_star+_1"
            g.message(meas)
            tmp_correlator = b2p._sigma_star_2pt(quark_prop_1, quark_prop_2, spm.T_mixed(), spm.Cgm())

            for p_n, p in enumerate(moms):
                g.message(mom_list[p_n])
                P = g.exp_ixp(p)
                correlator = g.slice(P * tmp_correlator, 3)
                for t, c in enumerate(correlator):
                    g.message(t, c)
                    corr[n_baryons * time_rev + baryon_n, p_n, t] = c

        elif(baryon_n == 1):
            meas = "Sigma+_4"
            g.message(meas)
            tmp_correlator = b2p._proton_2pt(quark_prop_1, quark_prop_2, spm.T_unpol(), spm.Cg5())

            for p_n, p in enumerate(moms):
                g.message(mom_list[p_n])
                P = g.exp_ixp(p)
                correlator = g.slice(P * tmp_correlator, 3)
                for t, c in enumerate(correlator):
                    g.message(t, c)
                    corr[n_baryons * time_rev + baryon_n, p_n, t] = c

        elif(baryon_n == 2):
            meas = "Lambda_4"
            g.message(meas)
            tmp_correlator = b2p._lambda_naive_2pt(quark_prop_1, quark_prop_2, spm.T_unpol(), spm.Cg5())

            for p_n, p in enumerate(moms):
                g.message(mom_list[p_n])
                P = g.exp_ixp(p)
                correlator = g.slice(P * tmp_correlator, 3)
                for t, c in enumerate(correlator):
                    g.message(t, c)
                    corr[n_baryons * time_rev + baryon_n, p_n, t] = c

        elif(baryon_n == 3):
            meas = "Proton_Polx"
            g.message(meas)
            tmp_correlator = b2p._proton_2pt(quark_prop_1, quark_prop_2, spm.T_polx(), spm.Cg5())

            for p_n, p in enumerate(moms):
                g.message(mom_list[p_n])
                P = g.exp_ixp(p)
                correlator = g.slice(P * tmp_correlator, 3)
                for t, c in enumerate(correlator):
                    g.message(t, c)
                    corr[n_baryons * time_rev + baryon_n, p_n, t] = c

        elif(baryon_n == 4):
            meas = "Proton_Poly"
            g.message(meas)
            tmp_correlator = b2p._proton_2pt(quark_prop_1, quark_prop_2, spm.T_poly(), spm.Cg5())

            for p_n, p in enumerate(moms):
                g.message(mom_list[p_n])
                P = g.exp_ixp(p)
                correlator = g.slice(P * tmp_correlator, 3)
                for t, c in enumerate(correlator):
                    g.message(t, c)
                    corr[n_baryons * time_rev + baryon_n, p_n, t] = c

        elif(baryon_n == 5):
            meas = "Proton_Polz"
            g.message(meas)
            tmp_correlator = b2p._proton_2pt(quark_prop_1, quark_prop_2, spm.T_polz(), spm.Cg5())

            for p_n, p in enumerate(moms):
                g.message(mom_list[p_n])
                P = g.exp_ixp(p)
                correlator = g.slice(P * tmp_correlator, 3)
                for t, c in enumerate(correlator):
                    g.message(t, c)
                    corr[n_baryons * time_rev + baryon_n, p_n, t] = c

        elif(baryon_n == 6):
            meas = "Lambda_octet"
            g.message(meas)
            tmp_correlator = b3p._lambda8_2pt(quark_prop_1, quark_prop_2, quark_prop_3, spm.T_unpol(), spm.Cg5())

            for p_n, p in enumerate(moms):
                g.message(mom_list[p_n])
                P = g.exp_ixp(p)
                correlator = g.slice(P * tmp_correlator, 3)
                for t, c in enumerate(correlator):
                    g.message(t, c)
                    corr[n_baryons * time_rev + baryon_n, p_n, t] = c



def baryon_spectrum(data_file, quark_prop_1, quark_prop_2, quark_prop_3, moms, mom_list, params):

    suN = params["su(n)"]
#    quarks_list = params["quarks"]
    kappa_list = params["kappa"]
    grid = quark_prop_1.grid
    Nt = grid.fdimensions[-1]
    if(suN == 2): n_baryons = 6
    elif(suN == 3): n_baryons = 7

    correlators = np.zeros((n_baryons * 2, len(mom_list), Nt), dtype = complex)
    time_rev = 0
    _baryon_loop(correlators, quark_prop_1, quark_prop_2, quark_prop_3, moms, mom_list, time_rev, n_baryons)

    #
    # Time reversed propagators
    #
    print("Doing the time reversed measurements")
    quark_prop_1 = g.eval(-g.gamma[5] * g.gamma["T"] * quark_prop_1 * g.gamma[5] * g.gamma["T"])
    quark_prop_2 = g.eval(-g.gamma[5] * g.gamma["T"] * quark_prop_2 * g.gamma[5] * g.gamma["T"])
    quark_prop_3 = g.eval(-g.gamma[5] * g.gamma["T"] * quark_prop_3 * g.gamma[5] * g.gamma["T"])
    time_rev = 1

    _baryon_loop(correlators, quark_prop_1, quark_prop_2, quark_prop_3, moms, mom_list, time_rev, n_baryons)
    _write_hdf5dset(correlators, data_file)
