#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2025
#
import gpt as g

g.default.set_verbose("random", False)
rng = g.random("benchmark", "vectorized_ranlux24_24_64")

evec = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
nevec = [tuple([-x for x in y]) for y in evec]
points = [(0, 0, 0, 0)] + evec + nevec
npoints = len(points)

for precision in [g.single, g.double]:
    grid = g.grid(g.default.get_ivec("--grid", [16, 16, 16, 32], 4), precision)
    N = g.default.get_int("--N", 1000)
    n_rhs = g.default.get_int("--n_rhs", 12)
    g.message(
        f"""


    9-point coarse operator with
    fdimensions  : {grid.fdimensions}
    n_rhs        : {n_rhs}
    precision    : {precision.__name__}
"""
    )

    # coarse grid stencil
    nbasis = g.default.get_int("--nbasis", 12)
    vcoarse = [g.vcomplex(grid, nbasis) for _ in range(n_rhs)]
    vdst = g.copy(vcoarse)
    g.message(f"Virtual blocks: {len(vcoarse[0].v_obj)}")
    nbasis_blocks = len(vcoarse[0].v_obj)
    mcoarse = [g.mcomplex(grid, nbasis) for _ in range(npoints)]
    rng.cnormal(mcoarse)
    rng.cnormal(vcoarse)

    otype = g.ot_vector_complex_additive_group(nbasis)
    cop = g.block.matrix_operator.compiled(
        {points[i] : mcoarse[i] for i in range(npoints)}
    )

    # Flops
    flops_per_matrix_vector_multiply = nbasis * (nbasis * 6 + (nbasis - 1) * 2)
    flops_per_vector_add = nbasis * 2
    flops_per_site = len(points) * flops_per_matrix_vector_multiply + (len(points) - 1) * flops_per_vector_add
    flops = flops_per_site * vcoarse[0].grid.gsites * N * n_rhs
    nbytes = (
        (len(points) * nbasis**2 * 2 + 2 * nbasis * 2)
        * precision.nbytes
        * vcoarse[0].grid.gsites
        * N
        * n_rhs
    )

    # Warmup
    for n in range(5):
        cop(vdst, vcoarse)

    # Time
    t0 = g.time()
    for n in range(N):
        cop(vdst, vcoarse)
    t1 = g.time()

    # Report
    GFlopsPerSec = flops / (t1 - t0) / 1e9
    GBPerSec = nbytes / (t1 - t0) / 1e9
    g.message(
        f"""
{N} applications of {nbasis} x {nbasis}, 9-point operator
    Time to complete            : {t1-t0:.2f} s
    Total performance           : {GFlopsPerSec:.2f} GFlops/s
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s"""
    )
