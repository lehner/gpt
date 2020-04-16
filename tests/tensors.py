#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

grid=g.grid([2,2,2,2], g.single)

cm=g.mcolor(grid)
cv=g.vcolor(grid)

cv[:]=0
cv[0,0,0,0,0]=1
cv[0,0,0,0,1]=2

cm @= cv * g.adj(cv)

g.message(cm)

res = g.eval(cv * g.adj(cv) * cm * cv)

g.message(res)


sc=g.vspincolor(grid)
sc[:]=0
sc[0,0,0,0]=g.vspincolor([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
scA=g.eval( g.gamma[0] * g.gamma[1] * sc)
scB=g.eval( g.gamma[0] * g.eval( g.gamma[1] * sc ) )
g.message(scA)
g.message(g.norm2(scA-scB))

