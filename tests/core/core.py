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
L = [16, 16, 16, 32]
grid_dp = g.grid(L, g.double)
grid_sp = g.grid(L, g.single)

# test fields
rng = g.random("test")
l_dp = rng.cnormal(g.vcolor(grid_dp))
l_sp = g.convert(l_dp, g.single)

################################################################################
# Test mview
################################################################################
c = g.coordinates(l_dp)
x = l_dp[c]
mv = g.mview(x)
assert mv.itemsize == 1 and mv.shape[0] == len(mv)
assert sys.getrefcount(x) == 3
del mv
assert sys.getrefcount(x) == 2


################################################################################
# Test assignments
################################################################################
pos = g.coordinates(l_dp)
lhs = g.lattice(l_dp)


def assign_copy():
    g.copy(lhs, l_dp)


def assign_pos():
    lhs[pos] = l_dp[pos]


def assign_pos_view():
    plan = g.copy_plan(lhs, l_dp)
    plan.destination += lhs.view[pos]
    plan.source += l_dp.view[pos]
    plan = plan()
    info = plan.info()
    for rank_dst, rank_src in info:
        assert rank_dst == rank_src
        assert rank_dst == lhs.grid.processor
        info_rank = info[(rank_dst, rank_src)]
        for index in info_rank:
            info_index = info_rank[index]
            # Make sure that after optimization only a single memcpy is needed
            assert info_index["blocks"] == 1
    plan(lhs, l_dp)


for method in [assign_copy, assign_pos, assign_pos_view]:
    lhs[:] = 0
    method()
    eps2 = g.norm2(lhs - l_dp) / g.norm2(l_dp)
    assert eps2 < 1e-25

################################################################################
# Test exp_ixp
################################################################################
# multiply momentum phase in l
p = 2.0 * np.pi * np.array([1, 2, 3, 4]) / L
exp_ixp = g.exp_ixp(p)

# Test one component
xc = (2, 3, 1, 5)
x = np.array(list(xc))
ref = np.exp(1j * np.dot(p, x)) * l_dp[xc]

val = g.eval(exp_ixp * l_dp)[xc]
eps = g.norm2(ref - val)
g.message("Reference value test: ", eps)
assert eps < 1e-25

# single/double
eps = g.norm2(exp_ixp * l_sp - g.convert(exp_ixp * l_dp, g.single)) / g.norm2(l_sp)
g.message("Momentum phase test single/double: ", eps)
assert eps < 1e-10

eps = g.norm2(g.inv(exp_ixp) * exp_ixp * l_dp - l_dp) / g.norm2(l_dp)
g.message("Momentum inverse test: ", eps)
assert eps < 1e-20

eps = g.norm2(g.adj(exp_ixp) * exp_ixp * l_dp - l_dp) / g.norm2(l_dp)
g.message("Momentum adj test: ", eps)
assert eps < 1e-20

eps = g.norm2(g.adj(exp_ixp * exp_ixp) * exp_ixp * exp_ixp * l_dp - l_dp) / g.norm2(
    l_dp
)
g.message("Momentum adj test (2): ", eps)
assert eps < 1e-20

################################################################################
# Test FFT
################################################################################
fft_l_sp = g.eval(g.fft() * l_sp)
eps = g.norm2(g.adj(g.fft()) * fft_l_sp - l_sp) / g.norm2(l_sp)
g.message("FFTinv * FFT:", eps)
assert eps < 1e-12

eps = g.norm2(g.sum(exp_ixp * l_sp) / np.prod(L) - fft_l_sp[1, 2, 3, 4])
g.message("FFT forward test:", eps)
assert eps < 1e-12

fft_mom_A = g.slice(
    g.exp_ixp(2.0 * np.pi * np.array([1, 2, 3, 0]) / L) * l_sp, 3
) / np.prod(L[0:3])
fft_mom_B = [g.vcolor(x) for x in g.eval(g.fft([0, 1, 2]) * l_sp)[1, 2, 3, 0 : L[3]]]
for t in range(L[3]):
    eps = g.norm2(fft_mom_A[t] - fft_mom_B[t])
    assert eps < 1e-12

################################################################################
# Test vcomplex
################################################################################
va = g.vcomplex(grid_sp, 30)
vb = g.lattice(va)
va[:] = g.vcomplex([1] * 15 + [0.5] * 15, 30)
vb[:] = g.vcomplex([0.5] * 5 + [1.0] * 20 + [0.2] * 5, 30)
va @= 0.5 * va + 0.5 * vb
assert abs(va[0, 0, 0, 0][3] - 0.75) < 1e-6
assert abs(va[0, 0, 0, 0][18] - 0.75) < 1e-6
assert abs(va[0, 0, 0, 0][28] - 0.35) < 1e-6

################################################################################
# MPI
################################################################################
grid_sp.barrier()
nodes = grid_sp.globalsum(1)
assert nodes == grid_sp.Nprocessors
a = np.array([[1.0, 2.0, 3.0], [4, 5, 6j]], dtype=np.complex64)
b = np.copy(a)
grid_sp.globalsum(a)
eps = a / nodes - b
assert np.linalg.norm(eps) < 1e-7


################################################################################
# Test Cshifts
################################################################################
# create a complex lattice on the grid
src = g.complex(grid_sp)

# zero out all points and set the value at global position 0,0,0,0 to 2
src[:] = 0
src[0, 0, 0, 0] = complex(2, 1)

# create a new lattice that is compatible with another
new = g.lattice(src)

# create a new lattice that is a copy of another
original = g.copy(src)

# or copy the contents from one lattice to another
g.copy(new, src)

# cshift into a new lattice dst
dst = g.cshift(src, 0, 1)
# dst[x] = src[x+1] -> src[0] == dst[15]
assert abs(dst[15, 0, 0, 0] - complex(2, 1)) < 1e-6

################################################################################
# Test multi inner_product
################################################################################
for grid in [grid_sp, grid_dp]:
    left = [g.vcomplex(grid, 8) for i in range(2)]
    right = [g.vcomplex(grid, 8) for i in range(4)]
    rng.cnormal([left, right])
    host_result = g.rank_inner_product(left, right, False)
    acc_result = g.rank_inner_product(left, right, True)
    eps = np.linalg.norm(host_result - acc_result) / np.linalg.norm(host_result)
    g.message(f"Test multi inner product host<>accelerator: {eps}")
    assert eps < 1e-13
    for i in range(2):
        for j in range(4):
            host_result_individual = g.rank_inner_product(left[i], right[j], False)
            acc_result_individual = g.rank_inner_product(left[i], right[j], True)
            eps = abs(host_result_individual - host_result[i, j]) / abs(
                host_result[i, j]
            )
            assert eps < 1e-13
            eps = abs(acc_result_individual - acc_result[i, j]) / abs(acc_result[i, j])
            assert eps < 1e-13
            if i == 0 and j == 0:
                ref = np.vdot(
                    left[i][:].astype(np.complex128), right[j][:].astype(np.complex128)
                )
                eps = abs(host_result_individual - ref) / abs(ref)
                assert eps < 1e-12

################################################################################
# Test multi linear_combination against expression engine
################################################################################
for grid in [grid_sp, grid_dp]:
    nbasis = 7
    nblock = 3
    nvec = 2
    basis = [g.vcomplex(grid, 8) for i in range(nbasis)]
    rng.cnormal(basis)
    dst = [g.vcomplex(grid, 8) for i in range(nvec)]
    coef = [[rng.cnormal() for i in range(nbasis)] for j in range(nvec)]
    # multi
    g.linear_combination(dst, basis, coef, nblock)
    for j in range(nvec):
        ref = g.vcomplex(grid, 8)
        ref[:] = 0
        for i in range(nbasis):
            ref += coef[j][i] * basis[i]
        eps2 = g.norm2(dst[j] - ref) / g.norm2(ref)
        g.message(f"Test linear combination of vector {j}: {eps2}")
        assert eps2 < 1e-13


################################################################################
# Test where
################################################################################
sel = g.complex(grid)
rng.uniform_int(sel, min=0, max=1)

yes = g.vcomplex(grid, 8)
no = g.vcomplex(grid, 8)
rng.cnormal([yes, no])

w = g.where(sel, yes, no)

eps = np.linalg.norm(w[:] - np.where(sel[:] != 0.0, yes[:], no[:])) / np.linalg.norm(
    w[:]
)
g.message(
    f"Test gpt.where <> numpy.where with a selection of {g.norm2(sel)} points: {eps}"
)
assert eps == 0.0


################################################################################
# Test comparators
################################################################################
a = g.complex(grid)
b = g.complex(grid)
rng.cnormal([a, b])

c = a < b
eps = np.linalg.norm(c[:] - (a[:] < b[:]).astype(np.int)) / np.linalg.norm(c[:])
g.message(f"Test a < b from gpt<>numpy: {eps}")
assert eps == 0.0

eps = g.norm2((b < a) - (a > b)) ** 0.5
g.message(f"Test a < b compatible with b > a: {eps}")
assert eps == 0.0
