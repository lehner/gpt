#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

g.default.set_verbose("random", False)
rng = g.random("benchmark")

# helper function
def relative_deviation(reference, result):
    diff = g.eval(reference - result)
    abs_dev = g.norm2(diff)
    rel_dev = abs_dev / g.norm2(reference)
    g.message(f"""absolute deviation = {abs_dev}, relative deviation = {rel_dev}""")
    return rel_dev


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

    # Lookup table
    mask_full = g.complex(fgrid)
    mask_full[:] = 1
    lut_full = g.lookup_table(cgrid, mask_full)

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

    # Warmup without lookup table
    for n in range(5):
        g.block.project(dst_default, src, basis)

    # Time without lookup table
    t0 = g.time()
    for n in range(N):
        g.block.project(dst_default, src, basis)
    t1 = g.time()

    # Warmup with lookup table
    for n in range(5):
        g.block.project_using_lut(dst_lut, src, basis, lut_full)

    # Time with lookup table
    t2 = g.time()
    for n in range(N):
        g.block.project_using_lut(dst_lut, src, basis, lut_full)
    t3 = g.time()

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

    # Report with lookuptable
    GFlopsPerSec = flops / (t3 - t2) / 1e9
    GBPerSec = nbytes / (t3 - t2) / 1e9
    g.message(
        f"""{N} applications of block_project_using_lut
    Time to complete            : {t3-t2:.2f} s
    Total performance           : {GFlopsPerSec:.2f} GFlops/s
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s
    """
    )

    # Verify agreement
    assert relative_deviation(dst_default, dst_lut) == 0.0
