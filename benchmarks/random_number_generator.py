#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Benchmark RNG
#
import gpt as g

g.default.set_verbose("random", False)

for engine in ["vectorized_ranlux24_24_64", "vectorized_ranlux24_389_64"]:
    rng = g.random("benchmark", engine)

    for precision in [g.single, g.double]:
        grid = g.grid(g.default.get_ivec("--grid", [16, 16, 16, 32], 4), precision)

        g.message(
            f"""

Benchmark RNG engine {engine} in {precision.__name__} precision

"""
        )

        for lattice in [g.complex, g.vspincolor, g.mspincolor]:
            # Source and destination
            dst = lattice(grid)

            # random source
            for i in range(3):
                t0 = g.time()
                rng.uniform_real(dst)
                t1 = g.time()
                gb = dst.global_bytes() / 1e9
                g.message(f"uniform_real({lattice.__name__}) iteration {i}: {gb/(t1-t0)} GB/s")

            g.message("")
