#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2024
#
# Benchmark Indexed Sum
#
import gpt as g
import numpy as np
import cgpt


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
Indexed Sum Benchmark with
    fdimensions  : {grid.fdimensions}
    precision    : {precision.__name__}
"""
    )

    # coordinates
    idx = g.complex(grid)
    idx[:] = np.ascontiguousarray(g.coordinates(idx)[:, 3].astype(np.complex128))

    # Source and destination
    for tp in [g.ot_singlet(), g.ot_vector_spin_color(4, 3), g.ot_matrix_spin_color(4, 3)]:
        for n in [1, 4, 16]:
            one = [g.lattice(grid, tp) for i in range(n)]
            rng.cnormal(one)

            # Rank inner product
            nbytes = (one[0].global_bytes() * n + idx.global_bytes()) * N

            # Time global
            dt = 0.0
            cgpt.timer_begin()
            for it in range(N + Nwarmup):
                if it >= Nwarmup:
                    dt -= g.time()
                ip = g.indexed_sum(one, idx, grid.gdimensions[3])
                if it >= Nwarmup:
                    dt += g.time()

            # Time local
            rank_dt = 0.0
            for it in range(N + Nwarmup):
                if it >= Nwarmup:
                    rank_dt -= g.time()
                g.slice(one, 3)
                if it >= Nwarmup:
                    rank_dt += g.time()


            # Report
            GBPerSec = nbytes / dt / 1e9
            rank_GBPerSec = nbytes / rank_dt / 1e9
            cgpt_t = g.timer("ris")
            cgpt_t += cgpt.timer_end()
            g.message(
                f"""{N} indexed_sum
    Object type                 : {tp.__name__}
    Block                       : {n}
    Time to complete            : {dt:.2f} s (indexed_sum)
    Time to complete            : {rank_dt:.2f} s (slice)
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s (indexed_sum)
    Effective memory bandwidth  : {rank_GBPerSec:.2f} GB/s (slice)

     {cgpt_t}
"""
                )

