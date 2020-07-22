#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

grid = g.grid([8, 8, 8, 8], g.single)
rng = g.random("test")

# outer product and inner product of tensors
lhs = g.vcolor([rng.cnormal() for i in range(3)])
rhs = g.vcolor([rng.cnormal() for i in range(3)])

outer = lhs * g.adj(rhs)
inner = g.adj(lhs) * rhs
inner_comp = 0.0
for i in range(3):
    inner_comp += lhs.array.conjugate()[i] * rhs.array[i]
    for j in range(3):
        assert abs(outer.array[i, j] - lhs.array[i] * rhs.array.conjugate()[j]) < 1e-14
assert abs(inner_comp - inner) < 1e-14

# TODO: the following is already implemented for vcomplex but should
# be implemented for all vectors
# cwise = lhs * rhs

# demonstrate slicing of internal indices
vc = g.vcomplex(grid, 30)
vc[0, 0, 0, 0, 0] = 1
vc[0, 0, 0, 0, 1:29] = 1.5
vc[0, 0, 0, 0, 29] = 2
vc_comp = g.vcomplex([1] + [1.5] * 28 + [2], 30)
eps2 = g.norm2(vc[0, 0, 0, 0] - vc_comp)
assert eps2 < 1e-13

# demonstrate mask
mask = g.complex(grid)
mask[:] = 0
mask[0, 1, 2, 3] = 1
vc[:] = vc[0, 0, 0, 0]
vcmask = g.eval(mask * vc)
assert g.norm2(vcmask[0, 0, 0, 0]) < 1e-13
assert g.norm2(vcmask[0, 1, 2, 3] - vc_comp) < 1e-13

# demonstrate sign flip needed for MG
sign = g.vcomplex([1] * 15 + [-1] * 15, 30)
vc_comp = g.vcomplex([1] + [1.5] * 14 + [-1.5] * 14 + [-2], 30)
vc @= sign * vc
eps2 = g.norm2(vc[0, 0, 0, 0] - vc_comp)
assert eps2 < 1e-13

# demonstrate matrix * vector
ntest = 30
mc_comp = g.mcomplex(
    [[rng.cnormal() for i in range(ntest)] for j in range(ntest)], ntest
)
mc = g.mcomplex(grid, ntest)
mc[:] = mc_comp
vc_comp = g.vcomplex([rng.cnormal() for i in range(ntest)], ntest)
vc = g.vcomplex(grid, ntest)
vc[:] = vc_comp
assert g.norm2(mc[0, 0, 0, 0] - mc_comp) < 1e-10

vc2 = g.eval(mc * vc)
vc2_comp = mc_comp * vc_comp
vc3 = g.eval(mc_comp * vc)
assert g.norm2(vc2[0, 0, 0, 0] - vc2_comp) < 1e-10
eps2 = g.norm2(vc2 - vc3) / g.norm2(vc2)
assert eps2 < 1e-10

# test transpose and adjoint of mcomplex
mc_adj = g.eval(g.adj(mc))
mc_transpose = g.eval(g.transpose(mc))
mc_array = mc[0, 0, 0, 0].array
mc_adj_array = mc_adj[0, 0, 0, 0].array
mc_transpose_array = mc_transpose[0, 0, 0, 0].array
assert np.linalg.norm(mc_adj_array - mc_array.transpose().conjugate()) < 1e-13
assert np.linalg.norm(mc_transpose_array - mc_array.transpose()) < 1e-13
assert g.norm2(g.adj(mc[0, 0, 0, 0]) - mc_adj[0, 0, 0, 0]) < 1e-25
assert g.norm2(g.transpose(mc[0, 0, 0, 0]) - mc_transpose[0, 0, 0, 0]) < 1e-25


# assign entire lattice
cm = g.mcolor(grid)
cv = g.vcolor(grid)
cv[:] = 0
cm[:] = 0

# assign position and tensor index
cv[0, 0, 0, 0, 0] = 1
cv[0, 0, 0, 0, 1] = 2

# read out entire tensor at position
assert g.norm2(cv[0, 0, 0, 0] - g.vcolor([1, 2, 0])) < 1e-13

# set three internal indices to a vector
cm[0, 0, 0, 0, [[0, 1], [2, 2], [0, 0]]] = g.vcolor([7, 6, 5])
assert g.norm2(cm[0, 0, 0, 0] - g.mcolor([[5, 7, 0], [0, 0, 0], [0, 0, 6]])) < 1e-13

# set center element for two positions
cm[[[0, 1, 0, 1], [1, 1, 0, 0]], 1, 2] = 0.4
cm[[[1, 1, 0, 0], [0, 1, 0, 1]], [[1, 1]]] = 0.5
assert g.norm2(cm[0, 1, 0, 1] - g.mcolor([[0, 0, 0], [0, 0.5, 0.4], [0, 0, 0]])) < 1e-13

# now test outer products
cm @= cv * g.adj(cv)
assert g.norm2(cm[0, 0, 0, 0] - g.mcolor([[1, 2, 0], [2, 4, 0], [0, 0, 0]])) < 1e-13

# test inner and outer products
res = g.eval(cv * g.adj(cv) * cm * cv)
eps2 = g.norm2(res - g.norm2(cv) ** 2.0 * cv)
assert eps2 < 1e-13

# create spin color matrix and peek spin index
msc = g.mspincolor(grid)
rng.cnormal(msc)

ms = g.mspin(grid)
mc = g.mcolor(grid)

# peek spin index 1,2
mc[:] = msc[:, :, :, :, 1, 2, :, :]

A = mc[0, 1, 0, 1]
B = msc[0, 1, 0, 1]
for i in range(3):
    for j in range(3):
        eps = abs(A[i, j] - B[1, 2, i, j])
        assert eps < 1e-13

mc[0, 1, 0, 1, 2, 2] = 5

# poke spin index 1,2
msc[:, :, :, :, 1, 2, :, :] = mc[:]

A = mc[0, 1, 0, 1]
B = msc[0, 1, 0, 1]
for i in range(3):
    for j in range(3):
        eps = abs(A[i, j] - B[1, 2, i, j])
        assert eps < 1e-13

# peek color
ms[:] = msc[:, :, :, :, :, :, 1, 2]

A = ms[0, 1, 0, 1]
B = msc[0, 1, 0, 1]
for i in range(4):
    for j in range(4):
        eps = abs(A[i, j] - B[i, j, 1, 2])
        assert eps < 1e-13

# gamma matrices applied to spin
vsc = g.vspincolor(grid)
vsc[:] = 0
vsc[0, 0, 0, 0] = g.vspincolor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
vscA = g.eval(g.gamma[0] * g.gamma[1] * vsc)
vscB = g.eval(g.gamma[0] * g.eval(g.gamma[1] * vsc))
assert g.norm2(vscA - vscB) < 1e-13

# set entire block to tensor
src = g.vspincolor(grid)
zero = g.vspincolor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
val = g.vspincolor([[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]])
src[:] = 0
src[:, :, :, 0] = val

for x in range(grid.fdimensions[0]):
    for t in range(grid.fdimensions[3]):
        compare = val if t == 0 else zero
        eps = g.norm2(src[x, 0, 0, t] - compare)
        assert eps < 1e-13

# spin and color traces
mc = g.eval(g.spin_trace(msc))
assert g.norm2(mc[0, 0, 0, 0] - g.spin_trace(msc[0, 0, 0, 0])) < 1e-13

ms = g.eval(g.color_trace(msc))
assert g.norm2(ms[0, 0, 0, 0] - g.color_trace(msc[0, 0, 0, 0])) < 1e-13

eps0 = g.norm2(g.trace(msc) - g.spin_trace(ms))
eps1 = g.norm2(g.trace(msc) - g.color_trace(mc))
assert eps0 < 1e-9 and eps1 < 1e-9

# create singlet by number
assert g.complex(0.5).array[0] == 0.5

# left and right multiplication of different data types with scalar
mc = g.mcomplex(grid, ntest)
for dti in [cv, cm, vsc, msc, vc, mc]:
    rng.cnormal(dti)
    eps = g.norm2(mask * dti - dti * mask)
    g.message(f"Done with {dti.otype.__name__}")
    assert eps == 0.0
