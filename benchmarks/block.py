#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#          Daniel Richtmann 2020
#
# Desc.: Benchmark blocking operators
#
import gpt as g
import numpy as np
import sys

g.default.set_verbose("random", False)
rng = g.random("benchmark")


for precision in [g.single, g.double]:
    fgrid = g.grid(g.default.get_ivec("--fgrid", [16, 16, 16, 32], 4), precision)
    cgrid = g.grid(g.default.get_ivec("--cgrid", [4, 4, 4, 8], 4), precision)
    N = g.default.get_int("--N", 1000)
    nbasis = g.default.get_int("--nbasis", 40)
    g.message(
        f"""
Lookup Table Benchmark with
    fine fdimensions    : {fgrid.fdimensions}
    coarse fdimensions  : {cgrid.fdimensions}
    precision           : {precision.__name__}
    nbasis              : {nbasis}
"""
    )

    # Source and destination
    src = g.vspincolor(fgrid)
    dst_default = g.vcomplex(cgrid, nbasis)
    dst_lut = g.vcomplex(cgrid, nbasis)

    # Basis
    basis = [g.vspincolor(fgrid) for i in range(nbasis)]

    # Randomize
    rng.cnormal(src)
    rng.cnormal(basis)

    # Make a plan (could give a non-trivial project mask as third argument to map)
    block_map = g.block.map(cgrid, basis)

    # Flops
    Nc = src.otype.shape[1]
    flops_per_site = (4 * Nc * 6 + (4 * Nc - 1) * 2) * nbasis
    flops = flops_per_site * src.grid.gsites * N
    nbytes = (
        (
            (nbasis * 2 * 4 * Nc + 2 * 4 * Nc) * src.grid.gsites
            + 2 * nbasis * dst_default.grid.gsites
        )
        * precision.nbytes
        * N
    )

    # Warmup
    for n in range(5):
        block_map.project(dst_default, src)

    # Time
    t0 = g.time()
    for n in range(N):
        block_map.project(dst_default, src)
    t1 = g.time()

    # Report without lookuptable
    GFlopsPerSec = flops / (t1 - t0) / 1e9
    GBPerSec = nbytes / (t1 - t0) / 1e9
    g.message(
        f"""{N} applications of block_project
    Time to complete            : {t1-t0:.2f} s
    Total performance           : {GFlopsPerSec:.2f} GFlops/s
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s
    """
    )
