#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import numpy as np
import h5py

# load configuration
U = g.load("/glurch/scratch/configs/cls/A653r000/cnfg/A653r000n1750")

# do everything in single-precision
#U = g.convert(U, g.single)

# use the gauge configuration grid
grid=U[0].grid
L = np.array(grid.fdimensions)

# quark
w=g.qcd.fermion.wilson_clover(U,{
    "kappa" : 0.137,
    "csw_r" : 0.,
    "csw_t" : 0.,
    "xi_0" : 1,
    "nu" : 1,
    "isAnisotropic" : False,
    "boundary_phases" : [ 1.0, 1.0, 1.0, -1.0 ]
})

# create point source
src=g.mspincolor(grid)
g.create.point(src, [0, 0, 0, 26])

# build solver using g5m and cg
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-15, "maxiter": 1000})

slv_eo2 = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

#propagator
dst=g.mspincolor(grid)
dst @= slv_eo2 * src

# momentum
moms=np.array(([-1,0,0,0], [0,-1,0,0], [0,0,-1,0],
[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0]), dtype=float)
mom=2.0*np.pi*moms/L
print("len(moms)", len(moms))

mom_list=[ "-100", "0-10", "00-1", "000", "100", "010", "001" ]

# list of mesons
data_file = 'results_a653/mesons2pt_double_eps15.h5'
Nt=48
Cg5=g.core.spin_matrices.spin_matrix.Cg5()

for p_n, p in enumerate(mom):
    g.message( "mom", mom_list[p_n] )
    P=g.exp_ixp(p)
    correlator = g.slice(g.trace( P * g.adj(dst) * dst), 3)

    for t,c in enumerate(correlator):
        g.message( t,c.real)
