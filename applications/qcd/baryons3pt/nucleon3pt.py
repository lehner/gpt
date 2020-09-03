#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#          Lorenzo Barca 2020
import gpt as g
import numpy as np
from gpt.qcd.spin_matrices import spin_matrix as spm

# load configuration
U = g.load("/glurch/scratch/configs/cls/A653r000/cnfg/A653r000n1750")

# do everything in single-precision
#U = g.convert(U, g.single)

# use the gauge configuration grid
grid = U[0].grid
L = np.array(grid.fdimensions)

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
cg = inv.cg({"eps": 1e-15, "maxiter": 1000})

slv_eo2 = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

#propagator
dst = g.mspincolor(grid)
dst @= slv_eo2 * src

Cg5 = spm.Cg5()
Polx = spm.T_polx()
Poly = spm.T_poly()
Polz = spm.T_polz()
Tunpol = spm.T_unpol()

pols = {"Polx": Polx, "Poly": Poly, "Polz": Polz, "Tunpol": Tunpol}

# momentum of the current
q = np.array(([-1,0,0,0], [0,-1,0,0],
[0,0,0,0], [1,0,0,0]), dtype=float)
q_mom = 2.0 * np.pi * q / L
g.message("len(moms)", len(q_mom))

q_list = [ "q_-100", "q_0-10", "q_000", "q_100" ]

sink_mom = 2.0 * np.pi * np.array(([-1,0,0,0], [0,0,0,0]), dtype=float)/L
sink_moms_list = ["sink_mom_-100", "sink_mom_000"]

t_sink = 31

for pol in pols:
    g.message(pol)
    pol_grp = nucl_grp.create_group(pol)

    # sequential source
    tmp_seq_src = g.qcd.sequential_source.nucleon3pt_seq_src.p2p_ubaru(dst, pols[pol], Cg5, t_sink)

    for snk_mom_n, snk_mom  in enumerate(sink_mom):
        g.message(sink_moms_list[snk_mom_n])
        P = g.exp_ixp(snk_mom)
        snk_mom_grp = pol_grp.create_group(sink_moms_list[snk_mom_n])

        seq_src = g.lattice(src)
        seq_src[:] = tmp_seq_src[:]
        seq_src @= P * seq_src

        sanity_corr = g.slice(g.trace(g.adj(seq_src) * seq_src), 3)
        g.message("Sequential Source Correlator")
        for t,c in enumerate(sanity_corr):
            g.message(t, c)

        # Sequential propagator
        seq_prop = g.lattice(src)
        seq_prop @= slv_eo2 * seq_src

        sanity_corr = g.slice(g.trace(g.adj(seq_prop) * seq_prop), 3)
        g.message("Sequential Propagator Correlator")
        for t,c in enumerate(sanity_corr):
            g.message(t, c)

        for q_n, q in enumerate(q_mom):
            g.message( q_list[q_n] )
            Q = g.exp_ixp(q)

            correlator = g.slice(g.trace(Q * g.gamma[5] * g.adj(seq_prop) * g.gamma[5] * \
                                         g.gamma["X"] * g.gamma[5] * dst), 3)

            for t,c in enumerate(correlator):
                g.message(t, c)


