#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Test small core features that are not sufficiently complex
#        to require a separate test file.  These tests need to be fast.
#
import gpt as g
import numpy as np
import sys, cgpt

# grid
L=[16,16,16,32]
grid_dp=g.grid(L, g.double)
grid_sp=g.grid(L, g.single)

# test fields
l_dp=g.random("test").cnormal(g.vcolor(grid_dp))
l_sp=g.convert( l_dp, g.single )

################################################################################
# Test mview
################################################################################
c=g.coordinates(l_dp)
x=l_dp[c]
mv=g.mview(x)
assert(mv.itemsize == 1 and mv.shape[0] == len(mv))
assert(sys.getrefcount(x) == 3)
del mv
assert(sys.getrefcount(x) == 2)


################################################################################
# Test exp_ixp
################################################################################
# multiply momentum phase in l
p=2.0*np.pi*np.array([ 1, 2, 3, 4 ]) / L
exp_ixp=g.exp_ixp(p)

# Test one component
xc=(2,3,1,5)
x=np.array(list(xc))
ref=np.exp(1j*np.dot(p,x)) * l_dp[xc]

val=g.eval( exp_ixp*l_dp )[xc]
eps=g.norm2(ref-val)
g.message("Reference value test: ",eps)
assert(eps<1e-25)

# single/double
eps=g.norm2( exp_ixp*l_sp - g.convert( exp_ixp*l_dp , g.single ) ) / g.norm2(l_sp)
g.message("Momentum phase test single/double: ",eps)
assert(eps < 1e-10)

eps=g.norm2(g.inv(exp_ixp) * exp_ixp * l_dp - l_dp) / g.norm2(l_dp)
g.message("Momentum inverse test: ",eps)
assert(eps < 1e-20)

eps=g.norm2(g.adj(exp_ixp)*exp_ixp*l_dp - l_dp) / g.norm2(l_dp)
g.message("Momentum adj test: ",eps)
assert(eps < 1e-20)

eps=g.norm2(g.adj(exp_ixp*exp_ixp)*exp_ixp*exp_ixp*l_dp - l_dp) / g.norm2(l_dp)
g.message("Momentum adj test (2): ",eps)
assert(eps < 1e-20)


################################################################################
# Test vcomplex
################################################################################
va=g.vcomplex(grid_sp,30)
vb=g.lattice(va)
va[:]=g.vcomplex([ 1 ] * 15 + [0.5] * 15,30)
vb[:]=g.vcomplex([ 0.5 ] * 5 + [ 1.0 ] * 20 + [0.2] * 5,30)
va @= 0.5 * va + 0.5 * vb
assert(abs(va[0,0,0,0][3] - 0.75) < 1e-6)
assert(abs(va[0,0,0,0][18] - 0.75) < 1e-6)
assert(abs(va[0,0,0,0][28] - 0.35) < 1e-6)

################################################################################
# MPI
################################################################################
grid_sp.barrier()
nodes=grid_sp.globalsum(1)
assert(nodes == grid_sp.Nprocessors)
a=np.array([ [ 1.0, 2.0, 3.0 ], [ 4,5,6j] ],dtype=np.complex64)
b=np.copy(a)
grid_sp.globalsum(a)
eps=a / nodes - b
assert(np.linalg.norm(eps) < 1e-7)




sys.exit(0)

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
