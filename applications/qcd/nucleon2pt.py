#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#          Lorenzo Barca 2020
import gpt as g
import numpy as np

# load configuration
U = g.load("/glurch/scratch/configs/cls/A653r000/cnfg/A653r000n1750")

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

Cg5 = g.qcd.spin_matrices.spin_matrix.Cg5()
Polx = g.qcd.spin_matrices.spin_matrix.T_polx()
Poly = g.qcd.spin_matrices.spin_matrix.T_poly()
Polz = g.qcd.spin_matrices.spin_matrix.T_polz()
Tunpol = g.qcd.spin_matrices.spin_matrix.T_unpol()

pols = {"Polx": Polx, "Poly": Poly, "Polz": Polz, "Tunpol": Tunpol}

di_quark = g.qcd.quarkContract.quarkContract13(g.eval(dst * Cg5), g.eval(Cg5 * dst) )

# momentum
moms = np.array(([-1,0,0,0], [0,-1,0,0], [0,0,-1,0],
[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0]), dtype=float)
mom = 2.0*np.pi*moms/L

mom_list = [ "mom_-100", "mom_0-10", "mom_00-1", "mom_000", "mom_100", "mom_010", "mom_001" ]

for pol in pols:
    pol_grp = nucl_grp.create_group(pol)
    for p_n, p in enumerate(mom):
        g.message("mom", mom_list[p_n])
        P = g.exp_ixp(p)
        correlator = g.slice(g.trace(P*pols[pol] * g.color_trace(dst * g.spin_trace(di_quark) ) ) + \
                             g.trace(P*pols[pol] * g.color_trace(dst * di_quark ) ), 3 )
        for t,c in enumerate(correlator):
            g.message(t, c)


