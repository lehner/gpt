#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Benchmark bandwidth of assignments
#
import gpt as g

# grid = g.grid([32,32,32,64], g.single)
grid = g.grid([8, 8, 8, 16], g.single)
# grid = g.grid([16, 16, 32, 128], g.single)
N = 10

rng = g.random("test")

for t in [g.mspincolor, g.vcolor, g.complex, g.mcolor]:
    lhs = t(grid)
    rhs = t(grid)
    rng.cnormal([lhs, rhs])

    # 2 * N for read/write
    GB = 2 * N * lhs.otype.nfloats * g.single.nbytes * grid.fsites / 1024.0 ** 3.0

    g.message(f"Test {lhs.otype.__name__}")

    t0 = g.time()
    for n in range(N):
        g.copy(lhs, rhs)
    t1 = g.time()
    g.message("%-50s %g GB/s" % ("Time for copy:", GB / (t1 - t0)))

    pos = g.coordinates(lhs)

    # create plan during first assignment, exclude from benchmark
    plan = g.copy_plan(lhs.view[pos], rhs.view[pos])

    t0 = g.time()
    for n in range(N):
        plan(lhs, rhs)
    t1 = g.time()
    g.message(
        "%-50s %g GB/s %g s" % ("Time for copy_plan:", GB / (t1 - t0), (t1 - t0) / N)
    )
