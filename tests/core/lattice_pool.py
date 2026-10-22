#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2026
#
# Lattice pool (--lattice_pool): the cgpt objects of deleted lattices are
# reused for new lattices.  Checked with the pool toggled at run time:
#   1) a workload of expressions, cshifts, checkerboards, gauge forces, and
#      reverse AD gives bitwise the same results with and without the pool,
#   2) repeating the workload with the pool creates no new cgpt lattices,
#   3) a reused lattice starts on the even checkerboard, and
#   4) a recreated grid reuses the pooled lattices of its predecessor.
#
import gpt as g
import gc
import gpt.core.lattice_pool as pool
import cgpt

use_pool = pool.use_pool

created = [0]
create_lattice = cgpt.create_lattice


def counting_create_lattice(*args):
    created[0] += 1
    return create_lattice(*args)


cgpt.create_lattice = counting_create_lattice

grid = g.grid([8, 8, 8, 8], g.double)
grid_eo = grid.checkerboarded(g.redblack)


def workload():
    rng = g.random("lattice_pool")
    res = []

    # expressions with new targets, in place, and with several v_obj
    a, b = rng.cnormal([g.mcolor(grid), g.mcolor(grid)])
    s = rng.cnormal(g.lattice(grid, g.ot_matrix_singlet(12)))
    assert len(s.v_obj) > 1
    res.append(g(a * b + 2 * g.adj(b)))
    res.append(g(g.trace(a * a * b)))
    res.append(g(s * s + g.adj(s)))
    a @= a * b
    res.append(a)
    res.append(g(g.cshift(a, 1, 1) - g.cshift(b, 2, -1)))

    # checkerboards
    v = rng.cnormal(g.vspincolor(grid))
    v_e, v_o = g.vspincolor(grid_eo), g.vspincolor(grid_eo)
    g.pick_checkerboard(g.even, v_e, v)
    g.pick_checkerboard(g.odd, v_o, v)
    w = g.lattice(v)
    g.set_checkerboard(w, g(2 * v_o))
    g.set_checkerboard(w, g(v_e + v_e))
    res.append(w)

    # gauge force and reverse AD of a functional of the links
    U = g.qcd.gauge.random(grid, rng)
    action = g.qcd.gauge.action.wilson(5.5)
    res += action.gradient(U, U)

    rad = g.ad.reverse
    x = rad.node(g.copy(U[0]))
    y = rad.node(g.copy(U[1]))
    c = g.sum(g.trace(x * y * g.adj(x) + y * y)).real
    c = rad.node(1.0) * c
    c()
    res += [g.copy(x.gradient), g.copy(y.gradient)]

    return res


def run(on):
    pool.use_pool = on
    r = workload()
    return [g.copy(x) for x in r] if on else r


reference = run(False)

for rep in range(3):
    # lattices in reference cycles (AD graphs) return with the collector
    gc.collect()
    created[0] = 0
    r = run(True)
    for x, y in zip(reference, r):
        assert x.otype.__name__ == y.otype.__name__
        assert x.checkerboard() is y.checkerboard()
        assert g.norm2(x - y) == 0.0
    g.message(f"repetition {rep}: {created[0]} new cgpt lattices, results bitwise equal")
    if rep > 0:
        # the copies of the results above were returned to the pool
        assert created[0] == 0
    del r, x, y

# a reused lattice starts on the even checkerboard
pool.use_pool = True
x = g.vspincolor(grid_eo)
x.checkerboard(g.odd)
del x
created[0] = 0
x = g.vspincolor(grid_eo)
assert created[0] == 0
assert x.checkerboard() is g.even
del x

# a recreated grid has the same cgpt object and reuses the pooled lattices
grid_s = g.grid([8, 8, 8, 8], g.single)
y = g.mcolor(grid_s)
o = y.v_obj[0]
del y, grid_s
grid_s = g.grid([8, 8, 8, 8], g.single)
created[0] = 0
y = g.mcolor(grid_s)
assert created[0] == 0 and y.v_obj[0] == o
del y, grid_s

n = sum(len(v) for v in pool.pool.values())
g.message(f"{n} lattices pooled")
pool.release_pool()
assert len(pool.pool) == 0
pool.use_pool = use_pool

g.message("All tests passed")
