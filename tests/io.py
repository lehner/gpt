#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

# load configuration
U = g.load("/hpcgpfs01/work/clehner/configs/16I_0p01_0p04/ckpoint_lat.IEEE64BIG.1100")

# Show metadata of field
g.message("Metadata", U[0].metadata)

def plaquette(U):
    # U[mu](x)*U[nu](x+mu)*adj(U[mu](x+nu))*adj(U[nu](x))
    tr=0.0
    vol=float(U[0].grid.gsites)
    for mu in range(4):
        for nu in range(mu):
            tr += g.sum( g.trace(U[mu] * g.cshift( U[nu], mu, 1) * g.adj( g.cshift( U[mu], nu, 1 ) ) * g.adj( U[nu] )) )
    return 2.*tr.real/vol/4./3./3.

# Calculate Plaquette
g.message(g.qcd.gauge.plaquette(U))
g.message(plaquette(U))

# Calculate U^\dag U
u = np.matrix(U[0][0,1,2,3])
g.message( u.getH() * u )

# Manipulate entire matrix and reassign to point in field
m=U[0][0,1,2,3]
g.message(m)
m[0][1] = 2
g.message(m)
U[0][0,1,2,3]=m
g.message(U[0][0,1,2,3])

# Expression including numpy array
r=g.eval( m*U[0] + U[1]*m )
g.message(g.norm2(r))

# test
tg=g.grid([2,2,2,2], g.single)
cf=g.complex(tg)
cf[:]=0
cf[0,0,0,0]=2
g.message(cf)

# todo: add explicit type conversion
# g.vspincolor([[0,1,-2],[0,1,0],[0,0,0],[0,0,0]])
g.message(g.eval(np.array([[0,1,-2],[0,1,0],[0,0,0],[0,0,0]])*cf))
