#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

# grid
grid=g.grid([2,2,2,2], g.single)

# test different lattice types
vc=g.vcolor(grid)
g.message(vc)

vz30=g.vcomplex(grid,30)
g.message(vz30)

vz30c=g.lattice(grid,vz30.describe())
vz30c[:]=g.vcomplex([ 1 ] * 15 + [0.5] * 15,30)
g.message(vz30c)

vz30b=g.lattice(vz30c)
vz30b[:]=g.vcomplex([ 0.5 ] * 5 + [ 1.0 ] * 20 + [0.2] * 5,30)

g.message(g.eval(vz30c + 0.3* vz30b))

# perform a barrier
grid.barrier()

# and a global sum over a number and a single-precision numpy array
nodes=grid.globalsum(1)

a=np.array([ [ 1.0, 2.0, 3.0 ], [ 4,5,6j] ],dtype=np.csingle)

grid.globalsum(a)

g.message(nodes,a)

# create a complex lattice on the grid
src=g.complex(grid)

# zero out all points and set the value at global position 0,0,0,0 to 2
src[:]=0
src[0,0,0,0]=complex(2,1)

# create a new lattice that is compatible with another
new=g.lattice(src)

# create a new lattice that is a copy of another
original=g.copy(src)

# or copy the contents from one lattice to another
g.copy(new,src)

# cshift into a new lattice dst
dst = g.cshift(src, 0, 1)

# show current memory usage
g.mem_report()
del original # free lattice and remove from scope
g.mem_report()

# or re-use an existing lattice object as target
g.cshift(dst, src, 0, 1)

# create a lattice expression
expr=g.trace(-(dst*dst) + 2*dst) + 0.5*(g.cshift(src, 0, 1)*dst + g.cshift(src, 0, -1))/3 - g.adj(dst + dst/2)

# and convert the expression to a new lattice or an existing one,
# later will allow for lists of expressions to be assigned to lists
# of target lattices
new=g.eval(expr)

# or re-use existing lattice object as target
g.eval(dst,expr)

# alternative notation
dst @= expr

# accumulate
dst+=expr

# print lattice
g.message(new)

# print adjungated field
g.message(g.eval(g.adj(new)))

# color matrix
cm=g.mcolor(grid)
cm[:]=0
cm[0,0,0,0,2,2]=1
cm[0,0,0,0,1,2]=1
g.message(cm)

g.message(g.eval(g.trace(cm)))

g.message(g.innerProductNorm2(src,src),g.norm2(src))
