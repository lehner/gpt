#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Benchmark Inner Products
#
import gpt as g
import cgpt


# helper functions to access data on host or accelerator
def access_host(a):
    for x in a:
        x.mview(g.host)


def access_accelerator(a):
    for x in a:
        x.mview(g.accelerator)


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
Inner Product Benchmark with
    fdimensions  : {grid.fdimensions}
    precision    : {precision.__name__}
"""
    )

    # Source and destination
    for tp in [g.ot_singlet(), g.ot_vector_spin_color(4, 3), g.ot_vector_singlet(12)]:
        for n in [1, 4]:
            one = [g.lattice(grid, tp) for i in range(n)]
            two = [g.lattice(grid, tp) for i in range(n)]
            rng.cnormal([one, two])

            # Rank inner product
            nbytes = (one[0].global_bytes() + two[0].global_bytes()) * N * n * n
            for use_accelerator, compute_name, access in [
                (False, "host", access_host),
                (True, "accelerator", access_accelerator),
            ]:
                # Time
                dt = 0.0
                cgpt.timer_begin()
                for it in range(N + Nwarmup):
                    access(one)
                    access(two)
                    if it >= Nwarmup:
                        dt -= g.time()
                    ip = g.rank_inner_product(one, two, use_accelerator)
                    if it >= Nwarmup:
                        dt += g.time()

                # Report
                GBPerSec = nbytes / dt / 1e9
                cgpt_t = g.timer("rip")
                cgpt_t += cgpt.timer_end()
                g.message(
                    f"""{N} rank_inner_product
    Object type                 : {tp.__name__}
    Block                       : {n} x {n}
    Data resides in             : {access.__name__[7:]}
    Performed on                : {compute_name}
    Time to complete            : {dt:.2f} s
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s

    {cgpt_t}
"""
                )
