#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2023
#
import gpt as g

g.default.set_verbose("random", False)
rng = g.random("benchmark", "vectorized_ranlux24_24_64")

for precision in [g.single, g.double]:
    grid = g.grid(g.default.get_ivec("--grid", [16, 16, 16, 32], 4), precision)
    N = g.default.get_int("--N", 1000)

    g.message(
        f"""
    Local Stencil Benchmark with
    fdimensions  : {grid.fdimensions}
    precision    : {precision.__name__}
"""
    )

    U = g.qcd.gauge.random(grid, rng, scale=0.5)
    _U = [1, 2, 3, 4]
    _X = 0
    _Xp = [1, 2, 3, 4]
    V = g.mcolor(grid)
    rng.element(V)
    # U = g.qcd.gauge.transformed(U, V)
    code = []
    for mu in range(4):
        for nu in range(0, mu):
            code.append(
                {
                    "target": 0,
                    "accumulate": -1 if (mu == 1 and nu == 0) else 0,
                    "weight": 1.0,
                    "factor": [
                        (_U[mu], _X, 0),
                        (_U[nu], _Xp[mu], 0),
                        (_U[mu], _Xp[nu], 1),
                        (_U[nu], _X, 1),
                    ],
                }
            )
    st = g.local_stencil.matrix(
        U[0], [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)], code
    )
    # test plaquette (local stencil, so no comms, only agrees on single rank)
    P = g.lattice(U[0])
    st(P, *U)
    # pl = g.sum(g.trace(P)).real / P.grid.gsites / 3 / 2 / 3
    # g.message(g.qcd.gauge.plaquette(U), pl)

    # Flops
    gauge_otype = U[0].otype
    Nc = gauge_otype.shape[0]
    flops_per_matrix_multiply = Nc**3 * 6 + (Nc - 1) * Nc**2 * 2
    flops_per_site = 3 * flops_per_matrix_multiply * 4 * 3
    flops = flops_per_site * P.grid.gsites * N
    nbytes = (5 * Nc * Nc * 2) * precision.nbytes * P.grid.gsites * N

    # Warmup
    for n in range(5):
        st(P, *U)

    # Time
    t0 = g.time()
    for n in range(N):
        st(P, *U)
    t1 = g.time()

    # Report
    GFlopsPerSec = flops / (t1 - t0) / 1e9
    GBPerSec = nbytes / (t1 - t0) / 1e9
    g.message(
        f"""{N} applications of plaquette stencil
    Time to complete            : {t1-t0:.2f} s
    Total performance           : {GFlopsPerSec:.2f} GFlops/s
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s"""
    )
