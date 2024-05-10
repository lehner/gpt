#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import numpy as np
import sys

rng = g.random("test")
vol = [16, 16, 16, 32]
grid_rb = g.grid(vol, g.single, g.redblack)
grid = g.grid(vol, g.single)
field = g.vcolor
Nlat = 12

################################################################################
# Spin/Color separation
################################################################################
msc = g.mspincolor(grid_rb)
rng.cnormal(msc)

xs = g.separate_spin(msc)
xc = g.separate_color(msc)

for s1 in range(4):
    for s2 in range(4):
        eps = np.linalg.norm(msc[0, 0, 0, 0].array[s1, s2, :, :] - xs[s1, s2][0, 0, 0, 0].array)
        assert eps < 1e-13

for c1 in range(3):
    for c2 in range(3):
        eps = np.linalg.norm(msc[0, 0, 0, 0].array[:, :, c1, c2] - xc[c1, c2][0, 0, 0, 0].array)
        assert eps < 1e-13


msc2 = g.lattice(msc)

g.merge_spin(msc2, xs)
assert g.norm2(msc2 - msc) < 1e-13

g.merge_color(msc2, xc)
assert g.norm2(msc2 - msc) < 1e-13

assert g.norm2(g.separate_color(xs[1, 2])[2, 0] - g.separate_spin(xc[2, 0])[1, 2]) < 1e-13


################################################################################
# Setup lattices
################################################################################
l_rb = [field(grid_rb) for i in range(Nlat)]
l = [field(grid) for i in range(Nlat)]
for i in range(Nlat):
    l_rb[i].checkerboard(g.odd)
rng.cnormal(l_rb)
rng.cnormal(l)


################################################################################
# Reverse lattice in time order in two ways and cross-check
################################################################################
def lattice_reverse_check(lat):
    grid = lat.grid
    l_rev_ref = g.lattice(lat)
    l_rev_ref.checkerboard(lat.checkerboard().inv())
    T = grid.fdimensions[3]
    for t in range(T):
        # 31 <- 0 => interchange even and odd sites
        l_rev_ref[:, :, :, T - t - 1] = lat[:, :, :, t]

    l_rev = g.merge(list(reversed(g.separate(lat, 3))), 3)
    eps = g.norm2(l_rev - l_rev_ref)
    g.message("Temporal inverse lattice test: ", eps)
    assert eps == 0.0


lattice_reverse_check(l[0])
lattice_reverse_check(l_rb[0])

################################################################################
# Test merge/separate here
################################################################################
assert all([g.norm2(x) > 0 for x in l])

# Test merging slices along a new last dimension 4 at a time
m = g.merge(l, N=4)
assert len(m) == Nlat // 4

for i in range(len(m)):
    for j in range(4):
        k = i * 4 + j
        assert g.norm2(l[k][1, 2, 0, 0] - m[i][1, 2, 0, 0, j]) == 0.0

# Test merging slices along a new 2nd dimension 4 at a time
m = g.merge(l, 1, N=4)
assert len(m) == Nlat // 4

for i in range(len(m)):
    for j in range(4):
        k = i * 4 + j
        assert g.norm2(l[k][1, 2, 0, 0] - m[i][1, j, 2, 0, 0]) == 0.0

test = g.separate(m, 1)

assert len(test) == Nlat
for i in range(len(l)):
    assert g.norm2(l[i] - test[i]) == 0.0

# default arguments should be compatible
test = g.separate(g.merge(l))
for i in range(len(l)):
    assert g.norm2(l[i] - test[i]) == 0.0

################################################################################
# Test split grid
################################################################################
for src in [l, l_rb]:
    split_grid = src[0].grid.split(
        g.default.get_ivec("--mpi_split", None, 4), src[0].grid.fdimensions
    )

    # perform this test without cache and with cache (to fill and then to test)
    cache = {}
    for it in range(3):
        for group_policy in [
            g.split_group_policy.separate,
            g.split_group_policy.together,
        ]:
            g.message(f"Iteration {it}, group_policy = {group_policy.__name__}")
            src_unsplit = [g.lattice(x) for x in src]
            t0 = g.time()
            src_split = g.split(src, split_grid, None if it == 0 else cache, group_policy)
            t1 = g.time()
            g.unsplit(src_unsplit, src_split, None if it == 0 else cache, group_policy)
            t2 = g.time()
            g.message(f"Timing: {t1-t0}s for split and {t2-t1}s for unsplit")
            for i in range(len(src)):
                eps2 = g.norm2(src_unsplit[i] - src[i]) / g.norm2(src[i])
                g.message(f"Split test {i} / {len(src)}: {eps2}")
                assert eps2 == 0.0


################################################################################
# Test scale per coordinate
################################################################################
grid_rb = g.grid([12, 8, 8, 8, 8], g.double, g.redblack)
a = rng.cnormal(g.vcolor(grid_rb))
b = g.lattice(a)
sc = np.array([rng.cnormal() for i in range(12)], np.complex128)
g.scale_per_coordinate(b, a, sc, 0)
a_s = g.separate(a, 0)
b_s = g.separate(b, 0)
for i in range(len(a_s)):
    eps2 = g.norm2(sc[i] * a_s[i] - b_s[i]) / g.norm2(b_s[i])
    assert eps2 < 1e-28
