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
from gpt.qcd.create_hdf5 import _check_meas, _check_propset, _write_hdf5dset_meson



def _meson_loop(corr, quark_prop_1, quark_prop_2, moms, mom_list, time_rev, n_mesons):

    gamma_sink = {"I":g.gamma["I"], "g1":g.gamma["X"], "g2":g.gamma["Y"], "g3":g.gamma["Z"], "g4":g.gamma["T"], \
                  "g1g5":g.eval(g.gamma["X"] * g.gamma[5]), "g2g5":g.eval(g.gamma["Y"] * g.gamma[5]), \
                  "g3g5":g.eval(g.gamma["Z"] * g.gamma[5]), "g4g5":g.eval(g.gamma["T"] * g.gamma[5]), \
                  "g5": g.gamma[5]}


    gamma_source = {"I":g.gamma["I"], "g1":g.gamma["X"], "g2":g.gamma["Y"], "g3":g.gamma["Z"], "g4":g.gamma["T"], \
                  "g1g5":g.eval(g.gamma["X"] * g.gamma[5]), "g2g5":g.eval(g.gamma["Y"] * g.gamma[5]), \
                  "g3g5":g.eval(g.gamma["Z"] * g.gamma[5]), "g4g5":g.eval(g.gamma["T"] * g.gamma[5]), \
                  "g5": g.gamma[5]}

    antiquark_prop2 = g.eval(g.gamma[5] * quark_prop_2 * g.gamma[5])
    meson_n = 0
    for gamma_sink_key, gamma_sink_matrix in gamma_sink.items():
        g.message(f"gamma_sink:{gamma_sink_key}")
        tmp_prop = g.eval(g.adj(antiquark_prop2) * gamma_sink_matrix * quark_prop_1)
        for gamma_source_key, gamma_source_matrix in gamma_source.items():
            g.message(f"gamma_source:{gamma_source_key}")
            tmp_correlator = g.trace(g.eval(tmp_prop * gamma_source_matrix))
            for p_n, p in enumerate(moms):
                g.message(mom_list[p_n])
                P = g.exp_ixp(p)
                correlator = g.slice(P * tmp_correlator, 3)
                for t, c in enumerate(correlator):
                    g.message(t, c)
                    corr[n_mesons * time_rev + meson_n, p_n, t] = c
            meson_n += 1


def meson_spectrum(data_file, light_quark_prop, strange_quark_prop, charm_quark_prop1, charm_quark_prop2, moms, mom_list, params):

    # check a new source measurement and copy the name
    tsrc_meas = _check_meas(data_file, "mesonspec")
    g.message(f"meas:{tsrc_meas}")
    kappa_list = params["kappa"]
    grid = light_quark_prop.grid
    Nt = grid.fdimensions[-1]
    n_mesons = 100

    quark_propagators = {"light":light_quark_prop, "strange":strange_quark_prop, 
                         "charm1":charm_quark_prop1, "charm2":charm_quark_prop2}

    for quark_flavour1 in ["light", "strange", "charm1", "charm2"]:
        for quark_flavour2 in ["light", "strange", "charm1", "charm2"]:
            # create flavour measurement
            prop_set = f"{quark_flavour1}_{quark_flavour2}"
            _check_propset(data_file, "mesonspec", tsrc_meas, prop_set)
            correlators = np.zeros((n_mesons * 2, len(mom_list), Nt), dtype = complex)
            time_rev = 0
            g.message(f"{prop_set} meson spectrum")
            quark_prop1 = quark_propagators[quark_flavour1]
            quark_prop2 = quark_propagators[quark_flavour2]

            _meson_loop(correlators, quark_prop1, quark_prop2, moms, mom_list, time_rev, n_mesons)
            g.mem_report()
            #
            # Time reversed propagators
            #
            print("Doing the time reversed measurements")
            quark_prop1 = g.eval(-g.gamma[5] * g.gamma["T"] * quark_prop1 * g.gamma[5] * g.gamma["T"])
            quark_prop2 = g.eval(-g.gamma[5] * g.gamma["T"] * quark_prop2 * g.gamma[5] * g.gamma["T"])
            time_rev = 1

            _meson_loop(correlators, quark_prop1, quark_prop2, moms, mom_list, time_rev, n_mesons)
            _write_hdf5dset_meson(correlators, data_file, tsrc_meas, prop_set)



