#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

# load configuration
for fine_grid, cb in [
    (g.grid([16, 12, 8, 16], g.single), None),
    (g.grid([16, 12, 8, 16], g.single, g.redblack), g.odd),
]:
    coarse_grid = g.grid([8, 4, 4, 8], fine_grid.precision)
    g.message(coarse_grid)

    # data types
    def vsc():
        return g.vspincolor(fine_grid)

    def vc12():
        return g.vcomplex(fine_grid, 12)

    # basis
    n = 30
    nvec = 2
    res = None
    tmpf_prev = None

    # test tensor projection
    left = g.vspincolor([[1, 1, 1]] * 2 + [[0, 0, 0]] * 2)  # 1 + g[5]
    right = g.vspincolor([[0, 0, 0]] * 2 + [[1, 1, 1]] * 2)  # 1 - g[5]
    rng = g.random("block_seed_string_13")
    basis = rng.cnormal([g.vspincolor(fine_grid) for i in range(n)])
    if cb is not None:
        for x in basis:
            x.checkerboard(cb)
    b = g.block.map(coarse_grid, basis)
    b.orthonormalize()
    b.orthonormalize()
    b = g.block.map(coarse_grid, basis + basis, tensor_projectors=[left] * n + [right] * n)
    fine_test = rng.cnormal(g.vspincolor(fine_grid))
    if cb is not None:
        fine_test.checkerboard(cb)
    test1 = g(b.project * fine_test)
    test2 = g(b.promote * test1)

    b = g.block.map(
        coarse_grid,
        [g(0.5 * x + 0.5 * g.gamma[5] * x) for x in basis]
        + [g(0.5 * x - 0.5 * g.gamma[5] * x) for x in basis],
    )
    test3 = g(b.project * fine_test)
    test4 = g(b.promote * test3)

    eps2 = g.norm2(test1 - test3) + g.norm2(test2 - test4)
    g.message("Chiral basis projector test:", eps2)
    assert eps2 < 1e-14

    # complete test
    for dtype in [vsc, vc12]:
        g.message(f"Data type {dtype.__name__}")
        basis = [dtype() for i in range(n)]
        if cb is not None:
            for x in basis:
                x.checkerboard(cb)
        rng = g.random("block_seed_string_13")
        rng.cnormal(basis)

        b = g.block.map(coarse_grid, basis)
        for i in range(2):
            g.message("Ortho step %d" % i)
            b.orthonormalize()

        # test coarse vector
        lcoarse = [g.vcomplex(coarse_grid, n) for i in range(nvec)]
        rng.cnormal(lcoarse)

        # report error of promote-project cycle
        lcoarse2 = g(b.project * b.promote * lcoarse)
        for i in range(nvec):
            lcoarse2_i = g(b.project * b.promote * lcoarse[i])
            eps2 = g.norm2(lcoarse2[i] - lcoarse2_i) / g.norm2(lcoarse2_i)
            g.message(eps2)
            assert eps2 < 1e-12
        err2 = g.norm2(lcoarse2[0] - lcoarse[0]) / g.norm2(lcoarse[0])
        g.message(f"Error^2 of promote-project cycle: {err2}")
        assert err2 < 1e-12

        # test project^dag == promote
        tfine = rng.cnormal(dtype())
        if cb is not None:
            tfine.checkerboard(cb)
        tcoarse = lcoarse[0]
        err = (
            abs(
                g.inner_product(tcoarse, b.project * tfine).conjugate()
                - g.inner_product(tfine, b.promote * tcoarse)
            )
            / tfine.grid.gsites**0.5
        )
        g.message(f"Test promote^dag == project: {err}")
        assert err < 1e-5

        # test block transfer on full grid
        if fine_grid.cb.n == 1:
            t = g.block.transfer(fine_grid, coarse_grid, basis[0].otype)

            bsum = t.sum(basis[0])
            fembed = t.embed(bsum)

            tcoarse = rng.cnormal(g.copy(bsum))
            err = (
                abs(
                    g.inner_product(tcoarse, t.sum * tfine).conjugate()
                    - g.inner_product(tfine, t.embed * tcoarse)
                )
                / tfine.grid.gsites**0.5
            )
            g.message(f"Test sum^dag == embed: {err}")
            assert err < 1e-5

            block_size = [fine_grid.gdimensions[i] // coarse_grid.gdimensions[i] for i in range(4)]
            for test_point in [(1, 2, 3, 4)]:
                comp_bsum = bsum[test_point]

                block_data = [
                    basis[0][
                        test_point[0] * block_size[0] + bx,
                        test_point[1] * block_size[1] + by,
                        test_point[2] * block_size[2] + bz,
                        test_point[3] * block_size[3] + bt,
                    ]
                    for bx in range(block_size[0])
                    for by in range(block_size[1])
                    for bz in range(block_size[2])
                    for bt in range(block_size[3])
                ]

                ref_bsum = block_data[0]
                for i in range(1, len(block_data)):
                    ref_bsum += block_data[i]

                err2 = g.norm2(comp_bsum - ref_bsum)
                g.message(f"Error^2 in sum: {err2}")
                assert err2 < 1e-12

                block_data = [
                    fembed[
                        test_point[0] * block_size[0] + bx,
                        test_point[1] * block_size[1] + by,
                        test_point[2] * block_size[2] + bz,
                        test_point[3] * block_size[3] + bt,
                    ]
                    for bx in range(block_size[0])
                    for by in range(block_size[1])
                    for bz in range(block_size[2])
                    for bt in range(block_size[3])
                ]

                err2 = 0.0
                for cc in block_data:
                    err2 += g.norm2(cc - ref_bsum) / len(block_data)
                g.message(f"Error^2 in embed: {err2}")
                assert err2 < 1e-12

# sublattices: restrict / prolong (strided picks with offsets and their
# adjoints), block sum / broadcast, also with a factor per direction and a
# coarse grid in another rank layout (with more than one rank)
fine_grid = g.grid([8, 4, 8, 16], g.double)
rng = g.random("sublattice")
fc = g.coordinates(fine_grid)
for factor, mpi in [(2, None), ([2, 1, 2, 4], None), (2, list(reversed(fine_grid.mpi)))]:
    f = [factor] * 4 if isinstance(factor, int) else factor
    coarse = None if mpi is None else g.grid([d // b for d, b in zip(fine_grid.fdimensions, f)], g.double, mpi=mpi)
    sl = g.block.sublattice(fine_grid, factor, coarse)
    cc = g.coordinates(sl.coarse_grid)
    for x, y in [
        (g.mcolor(fine_grid), g.mcolor(sl.coarse_grid)),
        (g.complex(fine_grid), g.complex(sl.coarse_grid)),
    ]:
        rng.cnormal([x, y])
        for offset in [(0, 0, 0, 0), (1, 0, 1, 3), (0, 0, 1, 2)]:
            if any(o >= b for o, b in zip(offset, f)):
                continue
            R, P = sl.restrict(offset), sl.prolong(offset)
            Rx = g(R * x)
            ref = x[np.ascontiguousarray(cc * np.array(f, dtype=cc.dtype) + np.array(offset, dtype=cc.dtype))]
            assert np.max(np.abs(Rx[cc] - ref)) == 0.0
            a, b = g.inner_product(Rx, y), g.inner_product(x, g(P * y))
            assert abs(a - b) < 1e-13 * abs(a)
            assert abs(g.norm2(g(P * y)) - g.norm2(y)) < 1e-13 * g.norm2(y)
        Sx = g(sl.sum * x)
        ref = g.copy(Sx)
        ref[:] = 0
        for offset in np.ndindex(*f):
            ref += sl.restrict(offset) * x
        err = g.norm2(Sx - ref) / g.norm2(ref)
        a, b = g.inner_product(Sx, y), g.inner_product(x, g(sl.broadcast * y))
        g.message(f"sublattice {factor} {sl.coarse_grid.mpi}, {x.otype.__name__}: block sum vs picks {err}, adjoint {abs(a - b) / abs(a)}")
        assert err < 1e-28 and abs(a - b) < 1e-13 * abs(a)

    # blocked links: typed like U, the products of the picked links
    U = g.qcd.gauge.random(fine_grid, rng)
    V = g.qcd.gauge.blocked_links(U, sl)
    for mu in range(4):
        assert V[mu].otype.__name__ == U[mu].otype.__name__
        ref = g.identity(V[mu])
        Us = U[mu]
        for k in range(f[mu]):
            ref = g(ref * (sl.restrict() * Us))
            Us = g.cshift(Us, mu, 1)
        assert g.norm2(V[mu] - ref) < 1e-26 * g.norm2(ref)
