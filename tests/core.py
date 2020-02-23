#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g

# gpt messaging system, only prints on g.rank() == 0
g.message("Using grid: ", g.default.grid)

# create a single precision grid with dimensions taken from "--grid ..."
grid=g.grid(g.default.grid, g.single)

# perform a barrier
grid.barrier()

# create a complex lattice on the grid
src=g.complex(grid)

# zero out all points and set the value at global position 0,0,0,0 to 1
src[:]=0
src[0,0,0,0]=2

# create a new lattice that is compatible with another
new=g.lattice(src)

# create a new lattice that is a copy of another
original=g.copy(src)

# or copy the contents from one lattice to another
g.copy(new,src)

# cshift into a new lattice dst
dst = g.cshift(src, 0, 1)

# show current memory usage
g.meminfo()
del original # free lattice and remove from scope
g.meminfo()

# or re-use an existing lattice object as target
g.cshift(dst, src, 0, 1)

# create a lattice expression
expr=-(dst*dst) + 2*dst + 0.5*(g.cshift(src, 0, 1)*dst + g.cshift(src, 0, -1))/3 - dst + dst/2

# and convert the expression to a new lattice or an existing one,
# later will allow for lists of expressions to be assigned to lists
# of target lattices
new=g.eval(expr)

# or re-use existing lattice object as target
g.eval(dst,expr)

# print lattice
g.message(new)
