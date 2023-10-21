#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2023
#
import gpt as g

g.default.set_verbose("random", False)
rng = g.random("benchmark", "vectorized_ranlux24_24_64")

evec = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
nevec = [tuple([-x for x in y]) for y in evec]

for precision in [g.single, g.double]:
    for fast_osites in [0, 1]:
        grid = g.grid(g.default.get_ivec("--grid", [16, 16, 16, 32], 4), precision)
        N = g.default.get_int("--N", 1000)

        g.message(
            f"""


    Local Stencil Benchmark with
    fdimensions  : {grid.fdimensions}
    precision    : {precision.__name__}
    fast_osites  : {fast_osites}
"""
        )

        # plaquette
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
        st = g.stencil.matrix(
            U[0],
            [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
            code,
        )
        st.memory_access_pattern(fast_osites=fast_osites)
        # test plaquette
        P = g.lattice(U[0])
        st(P, *U)
        pl = g.sum(g.trace(P)).real / P.grid.gsites / 3 / 2 / 3
        assert abs(g.qcd.gauge.plaquette(U) - pl) < precision.eps * 100

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
            f"""
{N} applications of plaquette stencil
    Time to complete            : {t1-t0:.2f} s
    Total performance           : {GFlopsPerSec:.2f} GFlops/s
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s"""
        )

        # covariant laplacian stencil
        src = g.vspincolor(grid)
        rng.cnormal(src)
        UdagShift = [g(g.adj(g.cshift(U[mu], mu, -1))) for mu in range(4)]
        _U = [0, 1, 2, 3]
        _UdagShift = [4, 5, 6, 7]
        _X = 0
        _Xp = [1, 2, 3, 4]
        _Xm = [5, 6, 7, 8]
        code = [(0, 1, _X, -1, -8.0, [])]
        for mu in range(4):
            code.append((0, 1, _Xp[mu], 0, 1.0, [(_U[mu], _X, 0)]))
            code.append((0, 1, _Xm[mu], 0, 1.0, [(_UdagShift[mu], _X, 0)]))
            # can switch last line to next one
            # code.append((0,1,_Xm[mu], 0, 1.0,[(_U[mu], _Xm[mu], 1)]))
        st = g.stencil.matrix_vector(U[0], src, [(0, 0, 0, 0)] + evec + nevec, code)
        st.memory_access_pattern(fast_osites=fast_osites)
        # test laplace
        dst = g.lattice(src)
        st(U + UdagShift, [dst, src])

        lap = g.create.smear.laplace(g.covariant.shift(U, boundary_phases=[1] * 4), [0, 1, 2, 3])
        dst2 = lap(src)
        eps2 = g.norm2(dst - dst2) / g.norm2(dst)
        assert eps2**0.5 < precision.eps * 100

        # Flops
        gauge_otype = U[0].otype
        Nc = gauge_otype.shape[0]
        Ns = 4
        flops_per_matrix_vector_multiply = Ns * Nc * (Nc * 6 + (Nc - 1) * 2)
        flops_per_vector_add = Ns * Nc * 2
        flops_per_site = 8 * flops_per_matrix_vector_multiply + 8 * flops_per_vector_add
        flops = flops_per_site * src.grid.gsites * N
        nbytes = (8 * Nc * Nc * 2 + Nc * Ns * 2) * precision.nbytes * src.grid.gsites * N

        # Warmup
        for n in range(5):
            st(U + UdagShift, [dst, src])

        # Time
        t0 = g.time()
        for n in range(N):
            st(U + UdagShift, [dst, src])
        t1 = g.time()

        # Report
        GFlopsPerSec = flops / (t1 - t0) / 1e9
        GBPerSec = nbytes / (t1 - t0) / 1e9
        g.message(
            f"""
{N} applications of laplace stencil
    Time to complete            : {t1-t0:.2f} s
    Total performance           : {GFlopsPerSec:.2f} GFlops/s
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s"""
        )
