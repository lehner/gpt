#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Benchmark bandwidth of assignments
#
import gpt as g

# grid = g.grid([32,32,32,64], g.single)
# grid = g.grid([8,8,8,16], g.single)
grid = g.grid([16, 16, 32, 128], g.single)
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

    pos = lhs.mview_coordinates()

    # create plan during first assignment, exclude from benchmark
    lhs[pos] = rhs.view[pos]

    t0 = g.time()
    for n in range(N):
        lhs[pos] = rhs.view[pos]
    t1 = g.time()
    g.message(
        "%-50s %g GB/s %g s"
        % ("Time for slice write (SIMD view):", GB / (t1 - t0), (t1 - t0) / N)
    )

    x = rhs[pos]
    lhs[pos] = x
    t0 = g.time()
    for n in range(N):
        lhs[pos] = x
    t1 = g.time()
    g.message("%-50s %g GB/s" % ("Time for slice write (SIMD layout):", GB / (t1 - t0)))

    break

    x = rhs[pos]  # cache
    t0 = g.time()
    for n in range(N):
        x = rhs[pos]
    t1 = g.time()
    g.message("%-50s %g GB/s" % ("Time for slice read (SIMD layout):", GB / (t1 - t0)))

    x = rhs[:]
    t0 = g.time()
    for n in range(N):
        x = rhs[:]
    t1 = g.time()
    g.message("%-50s %g GB/s" % ("Time for slice read:", GB / (t1 - t0)))

    lhs[:] = x
    t0 = g.time()
    for n in range(N):
        lhs[:] = x
    t1 = g.time()
    g.message("%-50s %g GB/s" % ("Time for slice write:", GB / (t1 - t0)))

    break

    t0 = g.time()
    for n in range(N):
        lhs @= rhs
    t1 = g.time()
    g.message("%-50s %g GB/s" % ("Time for assign:", GB / (t1 - t0)))

    t0 = g.time()
    lhs_mv = lhs.mview()
    rhs_mv = rhs.mview()
    sz = len(lhs_mv[0])
    for n in range(N):
        lhs_mv[0][0:sz] = rhs_mv[0][0:sz]
    t1 = g.time()
    g.message("%-50s %g GB/s" % ("Time for memoryview assign:", GB / (t1 - t0)))
