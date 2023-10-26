#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Benchmark Matrix Multiplication
#
import gpt as g

# mute random number generation
g.default.set_verbose("random", False)
rng = g.random("benchmark")

# main test loop
for precision in [g.single, g.double]:
    grid = g.grid(g.default.get_ivec("--grid", [16, 16, 16, 32], 4), precision)
    N = 10
    Nwarmup = 5
    g.message(
        f"""
Matrix Multiply Benchmark with
    fdimensions  : {grid.fdimensions}
    precision    : {precision.__name__}
"""
    )

    # Source and destination
    for tp in [g.ot_matrix_color(3), g.ot_matrix_spin(4), g.ot_matrix_spin_color(4, 3)]:
        one = g.lattice(grid, tp)
        two = g.lattice(grid, tp)
        three = g.lattice(grid, tp)
        rng.cnormal([one, two])

        # matrix multiply
        nbytes = 3.0 * one.global_bytes() * N
        n = (one.otype.nfloats // 2)**0.5
        flops_per_matrix_multiply = n * n * (n * 6 + (n - 1) * 2)
        flops = flops_per_matrix_multiply = grid.gsites * N * flops_per_matrix_multiply

        # Time
        dt = 0.0
        for it in range(N + Nwarmup):
            if it >= Nwarmup:
                dt -= g.time()
            g.eval(three, one * two)
            if it >= Nwarmup:
                dt += g.time()

        # Report
        GBPerSec = nbytes / dt / 1e9
        GFLPerSec = flops / dt / 1e9
        g.message(
            f"""{N} matrix_multiply
    Object type                 : {tp.__name__}
    Time to complete            : {dt:.2g} s
    GFlops/s                    : {GFLPerSec:.2f}
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s
"""
        )
