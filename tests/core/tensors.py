#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

grid=g.grid([8,8,8,8], g.single)
rng=g.random("test")

# demonstrate slicing of internal indices
vc=g.vcomplex(grid,30)
vc[0,0,0,0,0]=1
vc[0,0,0,0,1:29]=1.5
vc[0,0,0,0,29]=2
eps2=g.norm2(vc[0,0,0,0] - g.vcomplex([1] + [1.5]*28 + [2],30))
assert( eps2 < 1e-13 )

# assign entire lattice
cm=g.mcolor(grid)
cv=g.vcolor(grid)
cv[:]=0
cm[:]=0

# assign position and tensor index
cv[0,0,0,0,0]=1
cv[0,0,0,0,1]=2

# read out entire tensor at position
assert( g.norm2(cv[0,0,0,0] - g.vcolor([1,2,0])) < 1e-13 )

# set three internal indices to a vector
cm[0,0,0,0,[ [0,1], [2,2], [0,0] ]]=g.vcolor([7,6,5])
assert( g.norm2(cm[0,0,0,0] - g.mcolor([ [5,7,0], [0,0,0], [0,0,6] ])) < 1e-13 )

# set center element for two positions
cm[ [ [0,1,0,1],[1,1,0,0] ], 1, 2 ]=0.4
cm[ [ [1,1,0,0],[0,1,0,1] ], [ [1,1] ] ]=0.5
assert( g.norm2(cm[0,1,0,1] - g.mcolor([ [0,0,0],[0,0.5,0.4],[0,0,0] ])) < 1e-13 )

# now test outer products
cm @= cv * g.adj(cv)
assert( g.norm2(cm[0,0,0,0] - g.mcolor( [ [ 1, 2, 0], [2, 4, 0], [0,0,0] ] ) ) < 1e-13)

# test inner and outer products
res = g.eval(cv * g.adj(cv) * cm * cv)
eps2=g.norm2(res - g.norm2(cv)**2. * cv)
assert(eps2 < 1e-13)

# create spin color matrix and peek spin index
msc=g.mspin4color3(grid)
rng.cnormal(msc)

ms=g.mspin4(grid)
mc=g.mcolor3(grid)

# peek spin index 1,2
mc[:]=msc[:,:,:,:,1,2,:,:]

A=mc[0,1,0,1]
B=msc[0,1,0,1]
for i in range(3):
    for j in range(3):
        eps = abs(A.array[i,j] - B.array[1,2,i,j])
        assert(eps < 1e-13)

mc[0,1,0,1,2,2]=5

# poke spin index 1,2
msc[:,:,:,:,1,2,:,:]=mc[:]

A=mc[0,1,0,1]
B=msc[0,1,0,1]
for i in range(3):
    for j in range(3):
        eps = abs(A.array[i,j] - B.array[1,2,i,j])
        assert(eps < 1e-13)

# peek color
ms[:]=msc[:,:,:,:,:,:,1,2]

A=ms[0,1,0,1]
B=msc[0,1,0,1]
for i in range(4):
    for j in range(4):
        eps = abs(A.array[i,j] - B.array[i,j,1,2])
        assert(eps < 1e-13)

# gamma matrices applied to spin
sc=g.vspincolor(grid)
sc[:]=0
sc[0,0,0,0]=g.vspincolor([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
scA=g.eval( g.gamma[0] * g.gamma[1] * sc)
scB=g.eval( g.gamma[0] * g.eval( g.gamma[1] * sc ) )
assert(g.norm2(scA-scB) < 1e-13)

# set entire block to tensor
src=g.vspincolor(grid)
zero=g.vspincolor([[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
val=g.vspincolor([[1,1,1],[1,1,1],[0,0,0],[0,0,0]])
src[:]=0
src[:,:,:,0]=val

for x in range(grid.fdimensions[0]):
    for t in range(grid.fdimensions[3]):
        compare=val if t == 0 else zero
        eps=g.norm2( src[x,0,0,t] - compare )
        assert(eps < 1e-13)

# TODO: apply gamma to tensor


