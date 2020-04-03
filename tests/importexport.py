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
grid=g.grid([4,4,8,8],g.single)
src=g.vcolor(grid)
dst=g.vcolor(grid)
dst[:]=0

# fill a test lattice
for x in range(4):
    for y in range(4):
        for z in range(8):
            for t in range(8):
                src[x,y,z,t,0]=x + t*1j
                src[x,y,z,t,1]=y + t*1j
                src[x,y,z,t,2]=z + t*1j

# now create a random partition of this lattice distributed over all nodes
c=g.coordinates(grid)
random.seed(13)
for tr in range(10):
    shift=[ random.randint(0,8) for i in range(4) ]
    g.message(shift)
    for i in range(len(c)):
        for j in range(4):
            c[i][j] = (c[i][j] + shift[j]) % grid.gdimensions[j]
    data=src[c] # test global uniform memory system
    mvrestore=gpt.mview(data)
    err2=0.0
    for i,pos in enumerate(c):
        err2+=(data[i][0].real - pos[0])**2.0 + (data[i][1].real - pos[1])**2.0 + (data[i][2].real - pos[2])**2.0
    print(g.rank(),"found error",err2)
    dst[c]=mvrestore
    #data

    g.message(g.norm2(src-dst))

