#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

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
u = U[0][0,1,2,3]

v = g.vcolor([0,1,0])

g.message(g.adj(v))
g.message(g.adj(u) * u * v) 


gr=g.grid([2,2,2,2],g.single)
g.message(g.mspincolor(gr)[0,0,0,0] * g.vspincolor(gr)[0,0,0,0])

g.message(g.trace(g.mspincolor(gr)[0,0,0,0]))

# Expression including numpy array
r=g.eval( u*U[0] + U[1]*u )
g.message(g.norm2(r))

# test inner and outer products
v=g.vspincolor([[0,0,0],[0,0,2],[0,0,0],[0,0,0]])
w=g.vspincolor([[0,0,0],[0,0,0],[0,0,0],[1,0,0]])
xx=v * g.adj(w)
print(xx[1][3][2][0])
g.message(xx)
g.message(g.adj(v) * v)

