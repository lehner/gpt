#!/usr/bin/env python3
#
# Authors: Thomas Wurm 2021
#
import gpt
import numpy as np

# load configuration
rng = gpt.random("test")
L = [8, 8, 8, 16]
grid = gpt.grid(L, gpt.double)
u = gpt.qcd.gauge.random(grid, rng)
u_unit = gpt.qcd.gauge.unit(grid)

tol = (grid.precision.eps * 10) ** 2.0
gpt.message(f"tolerance: {tol}")

for t, t_type in [[gpt.vspincolor, "fermion"], [gpt.mspincolor, "propagator"]]:
    # initialize fields
    src = t(grid)
    dst_fermop = t(grid)
    dst_python = t(grid)

    for dimensions in [
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 3],
    ]:

        # compare against pure python implementation
        rng.cnormal(src)

        # initialize laplacians
        laplace_fermop = gpt.qcd.fermion.covariant_laplacian(
            u, dimensions=dimensions, boundary_phases=[1.0, 1.0, 1.0, -1.0]
        )
        laplace_python = gpt.create.smear.laplace(
            gpt.covariant.shift(
                u,
                {
                    "boundary_phases": [1.0, 1.0, 1.0, -1.0],
                },
            ),
            dimensions,
        )

        # apply laplacians
        dst_fermop @= gpt(laplace_fermop * src)
        dst_python @= gpt(laplace_python * src)

        eps2 = gpt.norm2(gpt.eval(dst_fermop - dst_python)) / gpt.norm2(dst_python)

        gpt.message(
            f"Laplacian compare to python with {t_type} for dimensions {dimensions}: {eps2}"
        )
        assert eps2 < tol

        # compare against exp_ixp(constant source)
        src[:] = gpt.vcolor([1, 0, 0])
        p = 2.0 * np.pi * np.array([1, 2, 3, 4.5]) / L
        src @= gpt(gpt.exp_ixp(p) * src)

        # initialize laplacian
        laplace_fermop = gpt.qcd.fermion.covariant_laplacian(
            u_unit, dimensions=dimensions, boundary_phases=[1.0, 1.0, 1.0, -1.0]
        )

        # apply laplacian
        dst_fermop @= gpt(laplace_fermop * src)
        eps2 = gpt.norm2(
            sum([2.0 * (np.cos(p[ii]) - 1.0) for ii in dimensions]) * src - dst_fermop
        ) / gpt.norm2(dst_fermop)
        gpt.message(
            f"Laplacian compare to exp with {t_type} for dimensions {dimensions}: {eps2}"
        )
        assert eps2 < tol
