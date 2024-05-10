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
        n_rhs = g.default.get_int("--n_rhs", 12)
        g.message(
            f"""


    Local Stencil Benchmark with
    fdimensions  : {grid.fdimensions}
    n_rhs        : {n_rhs}
    precision    : {precision.__name__}
    fast_osites  : {fast_osites}
"""
        )

        # coarse grid stencil
        nbasis = g.default.get_int("--nbasis", 12)
        vcoarse = [g.vcomplex(grid, nbasis) for _ in range(n_rhs)]
        g.message(f"Virtual blocks: {len(vcoarse[0].v_obj)}")
        nbasis_blocks = len(vcoarse[0].v_obj)
        mcoarse = [g.mcomplex(grid, nbasis) for _ in range(9)]
        rng.cnormal(mcoarse)
        rng.cnormal(vcoarse)

        # reference solve
        vref = [g.lattice(v) for v in vcoarse]
        for i in range(n_rhs):
            vref[i] = g(mcoarse[0] * vcoarse[i])
            for mu in range(4):
                vref[i] += mcoarse[1 + 2 * mu] * g.cshift(vcoarse[i], mu, 1)
                vref[i] += mcoarse[2 + 2 * mu] * g.cshift(vcoarse[i], mu, -1)
        # now coarse grid dirac operator
        _X = 0
        _Xp = [1, 2, 3, 4]
        _Xm = [5, 6, 7, 8]
        _ID = 0
        _M = [1, 3, 5, 7]
        _Mdag = [2, 4, 6, 8]
        code = []
        for i in range(n_rhs):
            ioff = nbasis_blocks * i
            for iblock in range(nbasis_blocks):
                # todo block matrix multiply here
                for jblock in range(nbasis_blocks):
                    matrix_index = nbasis_blocks * jblock + iblock
                    code.append(
                        (
                            iblock + ioff,
                            nbasis_blocks * n_rhs + jblock + ioff,
                            _X,
                            -1 if jblock == 0 else iblock + ioff,
                            1.0,
                            [(matrix_index, _X, 0)],
                        )
                    )
                    for mu in range(4):
                        code.append(
                            (
                                iblock + ioff,
                                nbasis_blocks * n_rhs + jblock + ioff,
                                _Xp[mu],
                                iblock + ioff,
                                1.0,
                                [(nbasis_blocks**2 * _M[mu] + matrix_index, _X, 0)],
                            )
                        )
                        code.append(
                            (
                                iblock + ioff,
                                nbasis_blocks * n_rhs + jblock + ioff,
                                _Xm[mu],
                                iblock + ioff,
                                1.0,
                                [(nbasis_blocks**2 * _Mdag[mu] + matrix_index, _X, 0)],
                            )
                        )
        st = g.stencil.matrix_vector(
            mcoarse[0],
            vcoarse[0],
            [(0, 0, 0, 0)] + evec + nevec,
            code,
            len(code) // nbasis_blocks // n_rhs,
        )
        st.memory_access_pattern(fast_osites=fast_osites)
        vdst = [g.lattice(v) for v in vcoarse]
        st(mcoarse, vdst + vcoarse)
        for i in range(n_rhs):
            eps2 = g.norm2(vdst[i] - vref[i]) / g.norm2(vref[i])
            # g.message(f"Test {i}: {eps2}")
            assert eps2**0.5 < precision.eps * 100

        # Flops
        flops_per_matrix_vector_multiply = nbasis * (nbasis * 6 + (nbasis - 1) * 2)
        flops_per_vector_add = nbasis * 2
        flops_per_site = 9 * flops_per_matrix_vector_multiply + 8 * flops_per_vector_add
        flops = flops_per_site * vcoarse[0].grid.gsites * N * n_rhs
        nbytes = (
            (9 * nbasis**2 * 2 + 2 * nbasis * 2)
            * precision.nbytes
            * vcoarse[0].grid.gsites
            * N
            * n_rhs
        )

        # Warmup
        for n in range(5):
            st(mcoarse, vdst + vcoarse)

        # Time
        t0 = g.time()
        for n in range(N):
            st(mcoarse, vdst + vcoarse)
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
