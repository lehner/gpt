#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import numpy as np

# load configuration
rng = g.random("test")
L = [8, 8, 8, 16]
grid = g.grid(L, g.double)
U = g.qcd.gauge.random(grid, rng)
U_unit = g.qcd.gauge.unit(grid)
V = rng.element(g.mcolor(grid))

# Test covariance of gauss smearing operator
smear = g.create.smear.gauss(U, sigma=0.5, steps=3, dimensions=[0, 1, 2])
U_transformed = g.qcd.gauge.transformed(U, V)
smear_transformed = g.create.smear.gauss(U_transformed, sigma=0.5, steps=3, dimensions=[0, 1, 2])

src = g.mspincolor(grid)
rng.cnormal(src)
dst1 = g(V * smear * src)
dst2 = g(smear_transformed * V * src)
eps2 = g.norm2(dst1 - dst2) / g.norm2(dst1)
g.message(f"Covariance test: {eps2}")
assert eps2 < 1e-29

# Test smearing operator on point source over unit gauge field
for dimensions in [[0, 1, 2], [0, 1, 2, 3]]:
    for sigma, steps in [(0.5, 3), (0.16, 2)]:
        smear_unit = g.create.smear.gauss(U_unit, sigma=sigma, steps=steps, dimensions=dimensions)
        src = g.vcolor(grid)
        src[:] = g.vcolor([1, 0, 0])

        # anti-periodic boundary conditions in time mean space and time are
        # differently quantized
        p = 2.0 * np.pi * np.array([1, 2, 3, 4.5]) / L
        src_mom = g(g.exp_ixp(p) * src)

        laplace = sum([2.0 * (np.cos(p[i]) - 1.0) for i in dimensions])
        factor = (1.0 + laplace * sigma**2.0 / steps / 4.0) ** steps
        dst = g(smear_unit * src_mom)
        eps2 = g.norm2(dst - factor * src_mom) / g.norm2(dst)
        g.message(f"Gaussian test using eigen representation: {eps2}")
        assert eps2 < 1e-29
