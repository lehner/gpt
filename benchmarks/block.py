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

N = g.default.get_int("--N", 1000)
nbasis = g.default.get_int("--nbasis", 40)
basis_n_block = g.default.get_int("--basis_n_block", 8)

for precision in [g.single, g.double]:
    fgrid = g.grid(g.default.get_ivec("--fgrid", [16, 16, 16, 32], 4), precision)
    cgrid = g.grid(g.default.get_ivec("--cgrid", [4, 4, 4, 8], 4), precision)

    # Basis
    basis = [g.vspincolor(fgrid) for i in range(nbasis)]

    # Make a plan
    block_map = g.block.map(cgrid, basis, basis_n_block=basis_n_block)

    # Randomize
    rng.cnormal(basis)

    for nvec in [1, 4]:
        g.message(
            f"""
Blocking Benchmark with
    fine fdimensions    : {fgrid.fdimensions}
    coarse fdimensions  : {cgrid.fdimensions}
    precision           : {precision.__name__}
    nbasis              : {nbasis}
    basis_n_block       : {basis_n_block}
    nvec                : {nvec}
"""
        )

        # Source and destination
        fine = [g.vspincolor(fgrid) for i in range(nvec)]
        coarse = [g.vcomplex(cgrid, nbasis) for i in range(nvec)]
        rng.cnormal(coarse)

        Nc = fine[0].otype.shape[1]
        fine_floats = fine[0].otype.nfloats
        fine_complex = fine_floats // 2
        coarse_floats = 2 * nbasis
        coarse_complex = coarse_floats // 2
        flops_per_cmul = 6
        flops_per_cadd = 2

        #######
        # Benchmark project
        #
        # Flops (count flops and bytes of nvec sequential operations as reference)
        flops_per_fine_site = (
            fine_complex * flops_per_cmul + (fine_complex - 1) * flops_per_cadd
        ) * nbasis
        flops = flops_per_fine_site * fine[0].grid.gsites * N * nvec
        nbytes = (
            (
                (nbasis + 1) * fine_floats * fine[0].grid.gsites  # read
                + coarse_floats * coarse[0].grid.gsites  # write
            )
            * precision.nbytes
            * N
            * nvec
        )

        # Warmup
        for n in range(5):
            block_map.project(coarse, fine)

        # Time
        t0 = g.time()
        for n in range(N):
            block_map.project(coarse, fine)
        t1 = g.time()

        # Report
        GFlopsPerSec = flops / (t1 - t0) / 1e9
        GBPerSec = nbytes / (t1 - t0) / 1e9
        g.message(
            f"""{N} applications of block_project
            Time to complete            : {t1-t0:.2f} s
            Total performance           : {GFlopsPerSec:.2f} GFlops/s
            Effective memory bandwidth  : {GBPerSec:.2f} GB/s
            """
        )

        #######
        # Benchmark promote
        #
        # Flops (count flops and bytes of nvec sequential operations as reference)
        flops_per_fine_site = (
            fine_complex * nbasis * flops_per_cmul + fine_complex * (nbasis - 1) * flops_per_cadd
        )
        flops = flops_per_fine_site * fine[0].grid.gsites * N * nvec
        nbytes = (
            (
                (nbasis + 1) * fine_floats * fine[0].grid.gsites  # read + write
                + coarse_floats * coarse[0].grid.gsites
            )
            * precision.nbytes
            * N
            * nvec
        )

        # Warmup
        for n in range(5):
            block_map.promote(fine, coarse)

        # Time
        t0 = g.time()
        for n in range(N):
            block_map.promote(fine, coarse)
        t1 = g.time()

        # Report
        GFlopsPerSec = flops / (t1 - t0) / 1e9
        GBPerSec = nbytes / (t1 - t0) / 1e9
        g.message(
            f"""{N} applications of block_promote
            Time to complete            : {t1-t0:.2f} s
            Total performance           : {GFlopsPerSec:.2f} GFlops/s
            Effective memory bandwidth  : {GBPerSec:.2f} GB/s
            """
        )
