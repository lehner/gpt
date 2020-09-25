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
from gpt.qcd.baryon_contractions import baryons_2prop
from gpt.qcd.baryon_contractions import baryons_3prop
import h5py

b2p = baryons_2prop()
b3p = baryons_3prop()


def baryon_spectrum(quark_prop_1, quark_prop_2, quark_prop_3, moms, mom_list, suN, data_file):
    Cg5  = spm.Cg5()
    Cgm = spm.Cgm()
    Cgm_NR = spm.Cgm_NR()
    Cg4m = spm.Cg4m()
    Cg5g4 = spm.Cg5g4()
    Cg5_NR = spm.Cg5_NR()
    Cg5_NR_negpar = spm.Cg5_NR_negpar()
    Polx = spm.T_polx()
    Poly = spm.T_poly()
    Polz = spm.T_polz()
    T_unpol = spm.T_unpol()
    T_mixed = spm.T_mixed()
    T_mixed_negpar = spm.T_mixed_negpar()

    grid = quark_prop_1.grid
    Nt = grid.fdimensions[-1]
    if(suN == 2): N_baryons = 20
    elif(suN == 3): N_baryons = 24

    with h5py.File(data_file, 'a') as hdf:
        for baryon_n in range(N_baryons):

            if(baryon_n== 0):
                meas = "Sigma+_1"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.proton_2pt(quark_prop_1, quark_prop_2, T_mixed, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)
                # add table ?

            elif(baryon_n== 1):
                meas = "Lambda_1"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b3p.lambda8_2pt(quark_prop_1, quark_prop_2, quark_prop_2, T_mixed, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 2):
                meas = "Sigma_star+_1"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.sigma_star_2pt(quark_prop_1, quark_prop_2, T_mixed, Cgm)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 3):
                meas = "Sigma+_2"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.proton_2pt(quark_prop_1, quark_prop_2, T_mixed, Cg5g4)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 4):
                meas = "Lambda_2"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b3p.lambda8_2pt(quark_prop_1, quark_prop_2, quark_prop_2, T_mixed, Cg5g4)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 5):
                meas = "Sigma_star+_2"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.sigma_star_2pt(quark_prop_1, quark_prop_2, T_mixed, Cg4m)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 6):
                meas = "Sigma+_3"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.proton_2pt(quark_prop_1, quark_prop_2, T_mixed, Cg5_NR)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 7):
                meas = "Lambda_3"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b3p.lambda8_2pt(quark_prop_1, quark_prop_2, quark_prop_2, T_mixed, Cg5_NR)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 8):
                meas = "Sigma_star+_3"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.sigma_star_2pt(quark_prop_1, quark_prop_2, T_unpol, Cgm_NR)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(4 * P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 9):
                meas = "Sigma+_4"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.proton_2pt(quark_prop_1, quark_prop_2, T_unpol, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 10):
                meas = "Sigma+_5"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.proton_2pt(quark_prop_1, quark_prop_2, T_unpol, Cg5g4)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 11):
                meas = "Sigma+_6"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.proton_2pt(quark_prop_1, quark_prop_2, T_unpol, Cg5_NR)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 12):
                meas = "Lambda_4"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.lambda_naive_2pt(quark_prop_1, quark_prop_2, T_unpol, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 13):
                meas = "Xi_1"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.xi_2pt(quark_prop_1, quark_prop_2, T_unpol, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 14):
                meas = "Lambda_5"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.lambda_naive_2pt(quark_prop_1, quark_prop_2, T_mixed, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)


            elif(baryon_n== 15):
                meas = "Xi_2"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.xi_2pt(quark_prop_1, quark_prop_2, T_mixed, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 16):
                meas = "Proton_negpar_3"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.proton_2pt(quark_prop_1, quark_prop_2, T_mixed_negpar, Cg5_NR_negpar)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 17):
                meas = "Proton_Polx"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.proton_2pt(quark_prop_1, quark_prop_2, Polx, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 18):
                meas = "Proton_Poly"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.proton_2pt(quark_prop_1, quark_prop_2, Poly, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 19):
                meas = "Proton_Polz"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.proton_2pt(quark_prop_1, quark_prop_2, Polz, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 20):
                meas = "Sigma_0"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b3p.sigma0_2pt(quark_prop_1, quark_prop_2, quark_prop_3, T_unpol, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 21):
                meas = "Lambda_octet"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b3p.lambda8_2pt(quark_prop_1, quark_prop_2, quark_prop_3, T_unpol, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 22):
                meas = "lambda8_to_sigma0"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b3p.lambda8_to_sigma0_2pt(quark_prop_1, quark_prop_2, quark_prop_3, T_unpol, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 23):
                meas = "sigma0_to_lambda8"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b3p.sigma0_to_lambda8_2pt(quark_prop_1, quark_prop_2, quark_prop_3, T_unpol, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n== 24):
                meas = "sigma_star+_3"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b3p.sigma_star_2pt(quark_prop_1, quark_prop_2, quark_prop_3, T_mixed, Cgm)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            else: g.message("Unknown baryon")
