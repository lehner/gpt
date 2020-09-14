#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#          Lorenzo Barca 2020
#   DRAFT
#
import gpt as g
import numpy as np
from gpt.qcd.spin_matrices import spin_matrix as spm
from gpt.qcd.baryon_spectrum import baryons_2prop
from gpt.qcd.baryon_spectrum import baryons_3prop
import h5py

b2p = baryons_2prop()
b3p = baryons_3prop()

# load configuration
U = g.load("/glurch/scratch/configs/cls/A653r000/cnfg/A653r000n1750")

# use the gauge configuration grid
grid = U[0].grid
L = np.array(grid.fdimensions)
Nt = 48

# quark
w = g.qcd.fermion.wilson_clover(U,{
    "kappa" : 0.137,
    "csw_r" : 0.,
    "csw_t" : 0.,
    "xi_0" : 1,
    "nu" : 1,
    "isAnisotropic" : False,
    "boundary_phases" : [ 1.0, 1.0, 1.0, -1.0 ]
})

# create point source
src = g.mspincolor(grid)
g.create.point(src, [0, 0, 0, 26])

# build solver using g5m and cg
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-10, "maxiter": 1000})

slv_eo2 = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

#propagator
dst = g.mspincolor(grid)
dst @= slv_eo2 * src


#quark_prop_1 = g.eval(- g.gamma[5] * g.gamma["T"] * dst * g.gamma[5] * g.gamma["T"])
#quark_prop_2 = g.eval(- g.gamma[5] * g.gamma["T"] * dst * g.gamma[5] * g.gamma["T"])
#quark_prop_3 = g.eval(- g.gamma[5] * g.gamma["T"] * dst * g.gamma[5] * g.gamma["T"])

quark_prop_1 = dst
quark_prop_2 = dst
quark_prop_3 = dst

Cg5 = spm.Cg5()
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


# momentum
moms = np.array(([-1,0,0,0], [0,-1,0,0], [0,0,-1,0],
[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0]), dtype=float)
mom = 2.0 * np.pi * moms / L

mom_list = [ "mom_-100", "mom_0-10", "mom_00-1", "mom_000", "mom_100", "mom_010", "mom_001" ]

data_file = 'baryons2pt.h5'
N_baryons = 24


with h5py.File(data_file, 'a') as hdf:
    for baryon_n in range(N_baryons):

        if( baryon_n== 0 ):
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

        elif( baryon_n== 1 ):
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

        elif( baryon_n== 2 ):
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

        elif( baryon_n== 3 ):
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

        elif( baryon_n== 4 ):
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

        elif( baryon_n== 5 ):
            meas = "Sigma_star+_2"
            g.message(meas)
#            hdf.create_group(meas)
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

        elif( baryon_n== 6 ):
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

        elif( baryon_n== 7 ):
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

        elif( baryon_n== 8 ):
            meas = "Sigma_star+_3"
            g.message(meas)
#            hdf.create_group(meas)
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
#           multiplied by 4

        elif( baryon_n== 9 ):
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

        elif( baryon_n== 10 ):
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

        elif( baryon_n== 11 ):
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

        elif( baryon_n== 12 ):
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

        elif( baryon_n== 13 ):
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

        elif( baryon_n== 14 ):
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


        elif( baryon_n== 15 ):
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

        elif( baryon_n== 16 ):
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

        elif( baryon_n== 17 ):
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

        elif( baryon_n== 18 ):
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

        elif( baryon_n== 19 ):
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

        elif( baryon_n== 20 ):
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

        elif( baryon_n== 21 ):
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

        elif( baryon_n== 22 ):
            meas = "lambda8_to_sigma0"
            g.message(meas)
#            hdf.create_group(meas)
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

        elif( baryon_n== 23 ):
            meas = "sigma0_to_lambda8"
            g.message(meas)
            hdf.create_group(meas)
            tmp_correlator = b3p.sigma0_to_lambda_2pt(quark_prop_1, quark_prop_2, quark_prop_3, T_unpol, Cg5)
            corr = np.zeros((len(mom_list), Nt), dtype = complex)

            for p_n, p in enumerate(moms):
                g.message(mom_list[p_n])
                P = g.exp_ixp(p)
                correlator = g.slice(P * tmp_correlator, 3)
                for t, c in enumerate(correlator):
                    g.message(t, c)
                    corr[p_n, t] = c
            hdf[meas].create_dataset('data', data = corr)

        elif( baryon_n== 24 ):
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

