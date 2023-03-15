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
