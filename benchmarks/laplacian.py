#!/usr/bin/env python3
#
# Authors: Thomas Wurm 2021
#
import gpt

# setup
rng = gpt.random("test")
N = gpt.default.get_int("--N", 1000)
vol = gpt.default.get_ivec("--grid", [16, 16, 16, 32], 4)

for precision in [gpt.single, gpt.double]:
    grid = gpt.grid(vol, precision)
    u = gpt.qcd.gauge.random(grid, rng)

    # initialize fields
    src = gpt.vspincolor(grid)
    rng.cnormal(src)
    dst = gpt.vspincolor(grid)

    for dimensions in [[0, 1, 2], [0, 1, 2, 3]]:
        # initialize laplacian
        laplacian = gpt.qcd.fermion.covariant_laplacian(
            u, dimensions=dimensions, boundary_phases=[1.0, 1.0, 1.0, -1.0]
        )

        # apply laplacian once
        laplacian(dst, src)

        # apply laplacian N times and measure time
        t0 = gpt.time()
        for _ in range(N):
            laplacian(dst, src)
        t1 = gpt.time()

        # calculate performance
        Nc = u[0].otype.shape[0]
        Nd = len(u)
        Ndis = 2 * len(dimensions)  # number of displacements

        flops_per_site = (
            Nc  # set result to (2 * N_dims) * src
            + Ndis * (2 * Nc ** 2 + Nc)  # multiply link and add
            + Nc  # write result
        )
        GFlopsPerSec = src.grid.gsites * N * Nd * flops_per_site / (t1 - t0) / 1e9

        nbytes_per_site = (
            (
                2 * Nc  # set result to (2 * N_dims) * src
                + Ndis * (Nc ** 2 + 2 * Nc)  # multiply link and add
                + 2 * Nc  # write result
            )
            * 2  # complex
            * precision.nbytes
        )
        GBPerSec = src.grid.gsites * N * Nd * nbytes_per_site / (t1 - t0) / 1e9

        gpt.message(
            f"""Laplacian benchmark {N} iterations and dims {dimensions}:
        Time to complete            : {t1-t0:.2f} s
        Precision                   : {precision.__name__}
        Total performance           : {GFlopsPerSec:.2f} GFlops/s
        Effective memory bandwidth  : {GBPerSec:.2f} GB/s"""
        )
