#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#          Daniel Richtmann 2020
#
# Desc.: Benchmark coarse operator
#
import gpt as g
import numpy as np
import cgpt

g.default.set_verbose("random", False)
rng = g.random("benchmark")

N = g.default.get_int("--N", 1000)
nbasis = g.default.get_int("--nbasis", 40)
level = g.default.get_int("--level", 0)

for precision in [g.single, g.double]:
    grid = g.grid(g.default.get_ivec("--grid", [6, 6, 6, 6], 4), precision)
    g.message(
        f"""
Coarse Operator Benchmark with
    fdimensions  : {grid.fdimensions}
    precision    : {precision.__name__}
    nbasis       : {nbasis}
    level        : {level}
"""
    )

    # Coarse operator
    A = [g.mcomplex(grid, nbasis) for __ in range(9)]
    rng.cnormal(A)
    co = g.qcd.fermion.coarse_fermion(A, {"level": level})

    # Source and destination
    src = g.vcomplex(grid, nbasis)
    dst = g.vcomplex(grid, nbasis)
    rng.cnormal(src)

    # Flops
    flops_per_site = 2 * nbasis * (36 * nbasis - 1)
    flops = flops_per_site * src.grid.gsites * N
    nbytes = (
        (9 * 2 * nbasis + 9 * 2 * nbasis * nbasis + 2 * nbasis)
        * precision.nbytes
        * src.grid.gsites
        * N
    )

    # Warmup
    # dst2 = g.copy(dst)
    # src2 = g.copy(src)
    for n in range(5):
        # co.mat([dst, dst2], [src, src2])
        co.mat(dst, src)

    # Time cgpt
    cgpt.timer_begin()
    for n in range(5):
        co.mat(dst, src)
    t_cgpt = g.timer("coarse_mat", True)
    t_cgpt += cgpt.timer_end()

    # Time
    t0 = g.time()
    for n in range(N):
        co.mat(dst, src)
    t1 = g.time()

    # Report
    GFlopsPerSec = flops / (t1 - t0) / 1e9
    GBPerSec = nbytes / (t1 - t0) / 1e9
    g.message(
        f"""{N} applications of M
    Time to complete            : {t1-t0:.2f} s
    Total performance           : {GFlopsPerSec:.2f} GFlops/s
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s

{t_cgpt}
"""
    )
