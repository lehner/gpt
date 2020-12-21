#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#          Daniel Richtmann 2020
#
# Benchmark Clover term
#
import gpt as g

g.default.set_verbose("random", False)
rng = g.random("benchmark")

for precision in [g.single, g.double]:
    grid = g.grid(g.default.get_ivec("--grid", [16, 16, 16, 32], 4), precision)
    N = g.default.get_int("--N", 1000)

    for use_legacy in [False, True]:
        g.message(
            f"""
    Clover Term Benchmark with
        fdimensions  : {grid.fdimensions}
        precision    : {precision.__name__}
        use_legacy   : {use_legacy}
    """
        )

        # Use Clover operator
        qc = g.qcd.fermion.wilson_clover(
            g.qcd.gauge.random(grid, rng, scale=0.5),
            {
                "kappa": 0.13565,
                "csw_r": 2.0171,
                "csw_t": 2.0171,
                "xi_0": 1,
                "nu": 1,
                "isAnisotropic": False,
                "boundary_phases": [1.0, 1.0, 1.0, 1.0],
                "use_legacy": use_legacy,
            },
        )

        # Source and destination
        src = g.vspincolor(qc.F_grid)
        dst = g.vspincolor(qc.F_grid)

        rng.cnormal(src)

        # Flops
        gauge_otype = qc.U[0].otype
        Nc = gauge_otype.shape[0]
        flops_per_site = 2 * ((8 - 2 / (2 * Nc)) * (2 * Nc) * (2 * Nc) - (2 * Nc) * 4)
        flops = flops_per_site * src.grid.gsites * N
        nbytes = (
            (2 * 4 * Nc + 2 * 2 * ((2 * Nc - 1) * Nc + Nc) + 2 * 4 * Nc)
            * precision.nbytes
            * src.grid.gsites
            * N
        )

        # Warmup
        for n in range(5):
            qc.Mooee.mat(dst, src)

        # Time
        t0 = g.time()
        for n in range(N):
            qc.Mooee.mat(dst, src)
        t1 = g.time()

        # Report
        GFlopsPerSec = flops / (t1 - t0) / 1e9
        GBPerSec = nbytes / (t1 - t0) / 1e9
        g.message(
            f"""{N} applications of Mooee
        Time to complete            : {t1-t0:.2f} s
        Total performance           : {GFlopsPerSec:.2f} GFlops/s
        Effective memory bandwidth  : {GBPerSec:.2f} GB/s"""
        )
