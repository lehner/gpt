#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys
import random
import cgpt

# load configuration
U = g.load("/hpcgpfs01/work/clehner/configs/32IDfine/ckpoint_lat.200")

# Show metadata of field
g.message("Metadata", U[0].metadata)

# to single precision
#U = g.convert(U, g.single)

# save in default gpt format
g.save("out",{ 
    "va\nl" : [0,1,3,"tes\n\0t",3.123456789123456789,1.123456789123456789e-7,1+3.1231251251234123413j], # fundamental data types
    "np" : g.coordinates(U[0].grid), # write numpy array from root node
    "U" : U # write list of lattices
})

# save in custom gpt format with different mpi distribution of local views
g.save("out2",{ 
    "val" : [0,1,3,"test",3.123456789123456789,1.123456789123456789e-7,1+3.1231251251234123413j], # fundamental data types
    "np" : g.coordinates(U[0].grid), # write numpy array from root node
    "U" : U # write list of lattices
},g.format.gpt({
    "mpi" : [ 2, 2, 2, 1 ] # save fields in 2 x 2 x 1 x 1 processor grid instead of --mpi grid
}))

#
# load function
#
# - g.load(fn)          loads everything in fn and creates new grids as needed
# - g.load(fn,{ "grids" : ..., "paths" :  ... })  both grids and paths are optional parameters and may be lists, 
#                                                 grids are re-used when loading, paths restricts which items to load (allows for glob.glob syntax /U/*)
res=g.load("out")

for i in range(4):
    g.message("Test first restore of U[%d]:" % i,g.norm2(res["U"][i] - U[i]))

res=g.load("out2",{ 
    "paths" : "/U/*"
})
for i in range(4):
    g.message("Test second restore of U[%d]:" % i,g.norm2(res["U"][i] - U[i]))

sys.exit(0)

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

# Precision change
Uf = g.convert(U, g.single)
g.message(g.qcd.gauge.plaquette(Uf))

Uf0 = g.convert(U[0], g.single)
g.message(g.norm2(Uf0))

del Uf0
g.mem_report()

# Slice
x=g.sum(Uf[0])

print(x)

grid=g.grid([4,4,4,4],g.single)
gr=g.complex(grid)

gr[0,0,0,0]=2
gr[1,0,0,0]=3

gride=g.grid([4,4,4,4],g.single,g.redblack)
gre=g.complex(gride)
g.pick_cb(g.even,gre,gr)
gre[2,0,0,0]=4
g.set_cb(gr,gre)
g.mem_report()


print(gre)

gre.checkerboard(g.odd)

print(gre)


sys.exit(0)

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
g.message(xx[1][3][2][0])
g.message(xx)
g.message(g.adj(v) * v)

g.message(g.transpose(v) * v)

u += g.adj(u)
g.message(u)


v=g.vspincolor([[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
l=g.vspincolor(gr)
l[:]=0
l[0,0,0,0]=v

g.message(l)

for mu in [ 0,1,2,3,5]:
    for nu in [0,1,2,3,5]:
        g.message(mu,nu,g.norm2(g.gamma[mu] * g.gamma[nu] * l + g.gamma[nu] * g.gamma[mu] * l)/g.norm2(l))

g.message(l)

m=g.mspincolor(gr)
m[0,0,0,0]=xx
m @= g.gamma[5] * m * g.gamma[5]
g.message(m)

