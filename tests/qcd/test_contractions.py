#!/usr/bin/env python3
#
# Authors:  Lorenzo Barca    2020
#           Christoph Lehner 2020
#
import gpt as g
import numpy as np

rng = g.random("test")
vol = [4, 4, 4, 8]
grid_rb = g.grid(vol, g.double, g.redblack)
grid = g.grid(vol, g.double)
field = g.vcolor
print(grid)

msc = g.mspincolor(grid)
rng.cnormal(msc)

# test quark contraction routine
xc=g.separate_color(msc)
a=0

for k in range(4):
    a += xc[1,1][2,2,2,2][k,k] * xc[2,2][2,2,2,2][2,3] - \
         xc[1,2][2,2,2,2][k,k] * xc[2,1][2,2,2,2][2,3] - \
         xc[2,1][2,2,2,2][k,k] * xc[1,2][2,2,2,2][2,3] + \
         xc[2,2][2,2,2,2][k,k] * xc[1,1][2,2,2,2][2,3]

'''
di_quark=g.core.spinmatr.quarkContract12( msc, msc )
g.message( "a=", a )
g.message( "di_quark:", di_quark[2,2,2,2][2,3,0,0])

eps = np.abs(( a - di_quark[2,2,2,2][2,3,0,0] )/ di_quark[2,2,2,2][2,3,0,0] )
g.message("eps: ", eps)

assert( eps < 1e-15 )

'''
msc1 = g.eval( g.gamma[5] * msc)

print(msc1[0,0,0,0][0,0,0,0])
