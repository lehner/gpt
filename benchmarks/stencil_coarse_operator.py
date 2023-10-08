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
    grid = g.grid(g.default.get_ivec("--grid", [16, 16, 16, 32], 4), precision)
    N = g.default.get_int("--N", 1000)

    g.message(
        f"""

        
    Local Stencil Benchmark with
    fdimensions  : {grid.fdimensions}
    precision    : {precision.__name__}
"""
    )

    # coarse grid stencil
    nbasis = g.default.get_int("--nbasis", 12)
    vcoarse = g.vcomplex(grid, nbasis)
    g.message(f"Virtual blocks: {len(vcoarse.v_obj)}")
    nbasis_blocks = len(vcoarse.v_obj)
    mcoarse = [g.mcomplex(grid, nbasis) for _ in range(9)]
    rng.cnormal(mcoarse)
    rng.cnormal(vcoarse)

    # reference solve
    vref = g.lattice(vcoarse)
    vref = g(mcoarse[0] * vcoarse)
    for mu in range(4):
        vref += mcoarse[1 + 2*mu] * g.cshift(vcoarse, mu, 1)
        vref += mcoarse[2 + 2*mu] * g.cshift(vcoarse, mu, -1)
    # now coarse grid dirac operator
    _X = 0
    _Xp = [1,2,3,4]
    _Xm = [5,6,7,8]
    _ID = 0
    _M = [1,3,5,7]
    _Mdag = [2,4,6,8]
    code = []
    for iblock in range(nbasis_blocks):
        # todo block matrix multiply here
        for jblock in range(nbasis_blocks):
            matrix_index = nbasis_blocks * jblock + iblock
            code.append((iblock,nbasis_blocks + jblock,_X,-1 if jblock == 0 else iblock,1.0,[(matrix_index,_X,0)]))
            for mu in range(4):
                code.append((iblock,nbasis_blocks + jblock,_Xp[mu], iblock, 1.0, [(nbasis_blocks**2 * _M[mu] + matrix_index, _X, 0)]))
                code.append((iblock,nbasis_blocks + jblock,_Xm[mu], iblock, 1.0, [(nbasis_blocks**2 * _Mdag[mu] + matrix_index, _X, 0)]))
    st = g.stencil.matrix_vector(
        mcoarse[0],
        vcoarse,
        [(0, 0, 0, 0)] + evec + nevec,
        code, len(code) // nbasis_blocks
    )
    vdst = g.lattice(vcoarse)
    st(mcoarse, [vdst,vcoarse])
    eps2 = g.norm2(vdst - vref) / g.norm2(vref)
    # g.message(f"Test: {eps2}")
    assert eps2 ** 0.5 < precision.eps * 100

    # Flops
    flops_per_matrix_vector_multiply = nbasis * (nbasis * 6 + (nbasis - 1) * 2)
    flops_per_vector_add = nbasis * 2
    flops_per_site = 9 * flops_per_matrix_vector_multiply + 8 * flops_per_vector_add
    flops = flops_per_site * vcoarse.grid.gsites * N
    nbytes = (9 * nbasis ** 2 * 2 + 2 * nbasis * 2) * precision.nbytes * vcoarse.grid.gsites * N

    # Warmup
    for n in range(5):
        st(mcoarse, [vdst,vcoarse])

    # Time
    t0 = g.time()
    for n in range(N):
        st(mcoarse, [vdst,vcoarse])
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
