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
    N = 100
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
        rng.cnormal([one, two])

        # Rank inner product
        nbytes = 3.0 * one.global_bytes() * N

        # Time
        dt = 0.0
        for it in range(N + Nwarmup):
            if it >= Nwarmup:
                dt -= g.time()
            g(one * two)
            if it >= Nwarmup:
                dt += g.time()

        # Report
        GBPerSec = nbytes / dt / 1e9
        g.message(
            f"""{N} matrix_multiply
    Object type                 : {tp.__name__}
    Time to complete            : {dt:.2f} s
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s
"""
        )