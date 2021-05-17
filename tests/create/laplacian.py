#!/usr/bin/env python3
#
# Authors: Thomas Wurm 2021
#
import gpt

# load configuration
rng = gpt.random("test")
L = [8, 8, 8, 16]
grid = gpt.grid(L, gpt.double)
u = gpt.qcd.gauge.random(grid, rng)

for t, t_type in [[gpt.vspincolor, "fermion"], [gpt.mspincolor, "propagator"]]:
    # initialize fields
    src = t(grid)
    dst_fermop = t(grid)
    dst_python = t(grid)
    rng.cnormal(src)

    for dimensions in [[0, 1, 2], [0, 1, 2, 3], [0, 1, 3]]:
        # initialize laplacians
        laplace_fermop = gpt.qcd.fermion.covariant_laplacian(u, dimensions=dimensions, boundary_phases=[1.0, 1.0, 1.0, -1.0])
        laplace_python = gpt.create.smear.laplace(gpt.covariant.shift(u, {"boundary_phases": [1.0, 1.0, 1.0, -1.0], }), dimensions)

        # apply laplacians
        dst_fermop @= gpt(laplace_fermop * src)
        dst_python @= gpt(laplace_python * src)

        eps2 = gpt.norm2(gpt.eval(dst_fermop - dst_python)) / gpt.norm2(dst_python)

        gpt.message(f"Laplacian test with {t_type} for dimensions {dimensions}: {eps2}")
        assert eps2 < 1e-30
