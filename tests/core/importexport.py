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

# add test for byte order swap
data = memoryview(bytearray(4))
data[:] = b"NUXI"
mdata = memoryview(bytes(4))
cgpt.munge_byte_order(mdata, data, 4)
assert mdata[::-1] == data


# import/export test
grid = g.grid([4, 4, 8, 8], g.single)
src = g.vcomplex(grid, 30)
dst = g.vcomplex(grid, 30)
dst[:] = 0

# fill a test lattice
for x in range(4):
    for y in range(4):
        for z in range(8):
            for t in range(8):
                src[x, y, z, t] = g.vcomplex([x + t * 1j, y + t * 1j, z + t * 1j] * 10, 30)

# now create a random partition of this lattice distributed over all nodes
c = (
    g.coordinates(grid).copy().view(np.ndarray)
)  # copy to make it writeable and lift local_coordinate type
random.seed(13)
for tr in range(10):
    shift = [random.randint(0, 8) for i in range(4)]
    for i in range(len(c)):
        for j in range(4):
            c[i][j] = (c[i][j] + shift[j]) % grid.gdimensions[j]
    data = src[c]  # test global uniform memory system
    mvrestore = g.mview(data)
    err2 = 0.0
    for i, pos in enumerate(c):
        for n in range(10):
            err2 += (
                (data[i][3 * n + 0].real - pos[0]) ** 2.0
                + (data[i][3 * n + 1].real - pos[1]) ** 2.0
                + (data[i][3 * n + 2].real - pos[2]) ** 2.0
            )
    dst[c] = mvrestore
    err2 = grid.globalsum(err2)
    err1 = g.norm2(src - dst)
    g.message("Test shift", tr, "/ 10 :", shift, "difference norm/e2:", err1, err2)
    assert err1 == 0.0 and err2 == 0.0
