#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Benchmark bandwidth of assignments
#
import gpt as g

grid = g.grid(g.default.get_ivec("--grid", [8, 8, 8, 16], 4), g.single)
N = g.default.get_int("--N", 10)

rng = g.random("test")

for t in [g.mspincolor, g.vcolor, g.complex, g.mcolor]:
    lhs = t(grid)
    rhs = t(grid)
    rng.cnormal([lhs, rhs])

    # 2 * N for read/write
    GB = 2 * N * lhs.global_bytes() / 1e9

    g.message(f"Test {lhs.otype.__name__}")

    # warmup
    g.copy(lhs, rhs)

    t0 = g.time()
    for n in range(N):
        g.copy(lhs, rhs)
    t1 = g.time()
    g.message("%-50s %g GB/s" % ("copy:", GB / (t1 - t0)))

    pos = g.coordinates(lhs)

    # create plan during first assignment, exclude from benchmark
    plan = g.copy_plan(lhs, rhs)
    plan.destination += lhs.view[pos]
    plan.source += rhs.view[pos]
    plan = plan()
    # g.message(plan.info())

    # warmup
    plan(lhs, rhs)

    t0 = g.time()
    for n in range(N):
        plan(lhs, rhs)
    t1 = g.time()
    g.message("%-50s %g GB/s %g s" % ("copy_plan:", GB / (t1 - t0), (t1 - t0) / N))


# spin/color separate/merge
msc = g.mspincolor(grid)
rng.cnormal(msc)

# 2 * N for read/write
GB = 2 * N * msc.otype.nfloats * grid.precision.nbytes * grid.fsites / 1e9

xc = g.separate_color(msc)
g.merge_color(msc, xc)
t0 = g.time()
for n in range(N):
    xc = g.separate_color(msc)
t1 = g.time()
for n in range(N):
    g.merge_color(msc, xc)
t2 = g.time()
g.message("%-50s %g GB/s %g s" % ("separate_color:", GB / (t1 - t0), (t1 - t0) / N))
g.message("%-50s %g GB/s %g s" % ("merge_color:", GB / (t2 - t1), (t2 - t1) / N))

xs = g.separate_spin(msc)
g.merge_spin(msc, xs)
t0 = g.time()
for n in range(N):
    xs = g.separate_spin(msc)
t1 = g.time()
for n in range(N):
    g.merge_spin(msc, xs)
t2 = g.time()
g.message("%-50s %g GB/s %g s" % ("separate_spin:", GB / (t1 - t0), (t1 - t0) / N))
g.message("%-50s %g GB/s %g s" % ("merge_spin:", GB / (t2 - t1), (t2 - t1) / N))
