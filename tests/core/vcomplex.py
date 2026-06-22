#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020, Mattia Bruno 2023
#
# Desc.: Illustrate core concepts and features
#
import gpt as g

L = [8, 4, 4, 4]
grid = g.grid(L, g.double)
rng = g.random("vcomplex")

def test_local_inner_product(v):
    i0 = g(g.adj(v) * v)
    i1 = g(g.trace(v * g.adj(v)))
    assert g.norm2(i0 - i1) < 1e-15

for N in [4, 10, 30]:
    v1 = g.vcomplex(grid, N)
    test_local_inner_product(v1)
    del v1
