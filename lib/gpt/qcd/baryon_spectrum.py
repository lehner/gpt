#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#          Lorenzo Barca 2020

import gpt as g
import numpy as np
from gpt.qcd import spin_matrices as spm
from gpt.qcd.baryon_contractions import baryons_2prop
from gpt.qcd.baryon_contractions import baryons_3prop
import h5py

b2p = baryons_2prop()
b3p = baryons_3prop()


def get_mom_list(moms):
    momentum_list = []
    for m in moms:
        p = 'mom_'
        for p_n, p_j in enumerate(m):
            if (p_n != 3):
                p += str(int(p_j))
        momentum_list.append(p)
    return momentum_list



def baryon_spectrum(data_file, moms, L, Nt, quark_prop_1, quark_prop_2, quark_prop_3):

    Cg5 = spm.spin_matrix.Cg5()
    Cgm = spm.spin_matrix.Cgm()
    Cg5g4 = spm.spin_matrix.Cg5g4()
    Polx = spm.spin_matrix.T_polx()
    Poly = spm.spin_matrix.T_poly()
    Polz = spm.spin_matrix.T_polz()
    T_unpol = spm.spin_matrix.T_unpol()
    T_mixed = spm.spin_matrix.T_mixed()
    T_mixed_negpar = spm.spin_matrix.T_mixed_negpar()

    mom_list = get_mom_list(moms)

    N_baryons = 22

    with h5py.File(data_file, 'a') as hdf:
        for baryon_n in range(N_baryons):

            if(baryon_n == 2):
                meas = "Sigma_star+_1"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.sigmast_2pt(quark_prop_1, quark_prop_2, T_mixed, Cgm)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)
                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)
#               add table

            elif(baryon_n == 9):
                meas = "Sigma+_4"
                g.message(meas)
                hdf.create_group(meas)
                tmp_correlator = b2p.sigma_2pt(quark_prop_1, quark_prop_2, T_unpol, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)

                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)


            elif(baryon_n == 12):
                meas = "Lambda_4"
                g.message(meas)
                hdf.create_group(meas)
                correlator = b2p.lambda_naive_2pt(quark_prop_1, quark_prop_2, T_unpol, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)

                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n == 17):
                meas = "Proton_Polx"
                g.message(meas)
                hdf.create_group(meas)
                correlator = b2p.sigma_2pt(quark_prop_1, quark_prop_2, Polx, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)

                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)


            elif(baryon_n == 18):
                meas = "Proton_Poly"
                g.message(meas)
                hdf.create_group(meas)
                correlator = b2p.sigma_2pt(quark_prop_1, quark_prop_2,Poly, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)

                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)


            elif(baryon_n == 19):
                meas = "Proton_Polz"
                g.message(meas)
                hdf.create_group(meas)
                correlator = b2p.sigma_2pt(quark_prop_1, quark_prop_2, Polz, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)

                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)


            elif(baryon_n == 21):
                meas = "Lambda_octet"
                g.message(meas)
                hdf.create_group(meas)
                correlator = b3p.lambda8_2pt(quark_prop_1, quark_prop_2, quark_prop_3, T_unpol, Cg5)
                corr = np.zeros((len(mom_list), Nt), dtype = complex)

                for p_n, p in enumerate(moms):
                    g.message(mom_list[p_n])
                    P = g.exp_ixp(p)
                    correlator = g.slice(P * tmp_correlator, 3)

                    for t, c in enumerate(correlator):
                        g.message(t, c)
                        corr[p_n, t] = c
                hdf[meas].create_dataset('data', data = corr)

            elif(baryon_n == 0): continue
            elif(baryon_n == 1): continue
            elif(baryon_n == 3): continue
            elif(baryon_n == 4): continue
            elif(baryon_n == 5): continue
            elif(baryon_n == 6): continue
            elif(baryon_n == 7): continue
            elif(baryon_n == 8): continue
            elif(baryon_n == 10): continue
            elif(baryon_n == 11): continue
            elif(baryon_n == 13): continue
            elif(baryon_n == 14): continue
            elif(baryon_n == 15): continue
            elif(baryon_n == 16): continue
            elif(baryon_n == 22): continue
            elif(baryon_n == 23): continue
            elif(baryon_n == 24): continue

            else: print("Unknown baryon")

