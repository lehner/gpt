#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2025
#
import gpt as g
import cgpt as c

g.default.set_verbose("random", False)
rng = g.random("benchmark", "vectorized_ranlux24_24_64")

max_point_sqr = g.default.get_int("--max_point_sqr", 1)

points = [
    (x, y, z, t)
    for x in range(-1, 2)
    for y in range(-1, 2)
    for z in range(-1, 2)
    for t in range(-1, 2)
    if x**2 + y**2 + z**2 + t**2 <= max_point_sqr
]
npoints = len(points)

N = g.default.get_int("--N", 1000)
n_rhs = g.default.get_int("--n_rhs", 12)
implementation = g.default.get("--implementation", None)
packed = g.default.get_int("--packed", 1)

for precision in [g.single, g.double]:
    grid = g.grid(g.default.get_ivec("--grid", [16, 16, 16, 32], 4), precision)
    g.message(
        f"""


    {npoints}-point coarse operator with
    fdimensions     : {grid.fdimensions}
    n_rhs           : {n_rhs}
    precision       : {precision.__name__}
    implementation  : {"default" if implementation is None else implementation}
    packed          : {packed}
"""
    )

    # coarse grid stencil
    nbasis = g.default.get_int("--nbasis", 12)
    if not packed:
        vcoarse = [g.vcomplex(grid, nbasis) for _ in range(n_rhs)]
        nbasis_blocks = len(vcoarse[0].v_obj)
    else:
        vcoarse = g.vcomplex(grid.inserted_dimension(0, n_rhs), nbasis)
        nbasis_blocks = len(vcoarse.v_obj)
    g.message(f"Virtual blocks: {nbasis_blocks}")
    vdst = g.copy(vcoarse)
    mcoarse = [g.mcomplex(grid, nbasis) for _ in range(npoints)]
    rng.cnormal(mcoarse)
    rng.cnormal(vcoarse)

    otype = g.ot_vector_complex_additive_group(nbasis)
    points_dictionary = {points[i]: mcoarse[i] for i in range(npoints)}
    cop = g.block.matrix_operator.compiled(
        points_dictionary,
        implementation=implementation,
        packed_right_hand_sides=None if packed == 0 else n_rhs,
    )
    if not packed:
        rcop = g.block.matrix_operator.compiled(points_dictionary, implementation="reference")

    # Flops
    flops_per_matrix_vector_multiply = nbasis * (nbasis * 6 + (nbasis - 1) * 2)
    flops_per_vector_add = nbasis * 2
    flops_per_site = (
        npoints * flops_per_matrix_vector_multiply + (npoints - 1) * flops_per_vector_add
    )
    flops = flops_per_site * grid.gsites * N * n_rhs
    nbytes = (npoints * nbasis**2 * 2 + 2 * nbasis * 2) * precision.nbytes * grid.gsites * N * n_rhs

    # Test
    if not packed:
        vref = g.copy(vdst)
        cop(vdst, vcoarse)
        rcop(vref, vcoarse)

        eps = (
            sum([g.norm2(vdst[i] - vref[i]) for i in range(n_rhs)])
            / sum([g.norm2(vref[i]) for i in range(n_rhs)])
        ) ** 0.5
        eps_ref = grid.precision.eps * 10
        g.message(f"Test implementation: {eps}")
        assert eps < eps_ref

    # Warmup
    for n in range(5):
        cop(vdst, vcoarse)

    # Time
    t0 = g.time()
    for n in range(N):
        if n == 10:
            c.profile_trigger(1)
        elif n == 40:
            c.profile_trigger(0)
        cop(vdst, vcoarse)
    t1 = g.time()

    # Report
    GFlopsPerSec = flops / (t1 - t0) / 1e9
    GBPerSec = nbytes / (t1 - t0) / 1e9
    g.message(
        f"""
{N} applications of {nbasis} x {nbasis}, {npoints}-point operator
    Time to complete            : {t1 - t0:.2g} s
    Total performance           : {GFlopsPerSec:.2f} GFlops/s
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s"""
    )
