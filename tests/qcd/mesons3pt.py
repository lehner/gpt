#!/usr/bin/env python3
#
# Authors: Lorenzo Barca    2020
#          Christoph Lehner 2020
#
import gpt as g
import numpy as np

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
g.create.point(src, [0,0,0,26])


# build cg solver
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-15, "maxiter": 1000})

slv_eo2 = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

#propagator
dst=g.mspincolor(grid)
dst @= slv_eo2 * src

# momentum
moms=np.array( ([-1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,0]), dtype=float)
mom=2.0*np.pi*moms/L
print("len(moms)", len(moms))
mom_list=[ "-100", "0-10", "00-1", "000" ]

# 2pt
correlator_2pt=g.slice(g.trace( dst * g.adj(dst) ), 3)

t_sink = 31

# currents at insertion time
currents=[ g.gamma["X"], g.gamma["Y"], g.gamma["Z"], g.gamma["T"] ]

for p_n, p in enumerate(moms):
    g.message( "a0_rhoX" )
    g.message( "mom ", mom_list[p_n] )
    P=g.exp_ixp(p)
    G_sink = g.gamma["X"]* P
    G_src = g.gamma["I"]

    # sequential source
    seq_src = g.lattice(dst)
    seq_src[:] = 0
    seq_src[:, :, :, t_sink] = dst[:, :, :, t_sink]
    seq_src @= G_sink * seq_src
    seq_src @= seq_src * G_src

    # sequential propagator
    seq_prop = g.lattice(seq_src)
    seq_prop @= slv_eo2 * seq_src

    for curr_n, curr in enumerate(currents):
        correlator_3pt = g.slice(g.trace( g.gamma[5] * g.adj(seq_prop) * g.gamma[5] * curr * dst ) , 3 )

	#  output
        for t in range( len(correlator_2pt) ) :
            g.message( t, correlator_3pt[t] )
