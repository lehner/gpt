#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

# load configuration
fine_grid = g.grid([8, 8, 8, 16], g.single)

# basis
n = 31
basis = [g.vcomplex(fine_grid, 30) for i in range(n)]
rng = g.random("block_seed_string_13")
rng.cnormal(basis)

# gram-schmidt
for i in range(n):
    basis[i] /= g.norm2(basis[i]) ** 0.5
    g.orthogonalize(basis[i], basis[:i])

    for j in range(i):
        eps = g.inner_product(basis[j], basis[i])
        g.message(" <%d|%d> =" % (j, i), eps)
        assert abs(eps) < 1e-6
