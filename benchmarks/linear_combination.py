#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2021
#
# Benchmark (bi)linear_combination
#
import gpt as g
import numpy as np

# mute random number generation
g.default.set_verbose("random", False)
rng = g.random("benchmark")

# main test loop
for precision in [g.single, g.double]:
    grid = g.grid(g.default.get_ivec("--grid", [16, 16, 16, 32], 4), precision)
    N = 100
    Nwarmup = 5
    g.message(
        f"""
Benchmark linear combination
    fdimensions  : {grid.fdimensions}
    precision    : {precision.__name__}
"""
    )

    # Source and destination
    for tp in [g.ot_matrix_color(3), g.ot_matrix_spin(4), g.ot_vector_spin_color(4, 3)]:
        for nbasis in [4, 8, 16]:
            basis = [g.lattice(grid, tp) for i in range(nbasis)]
            result = g.lattice(grid, tp)
            rng.cnormal(basis)

            # Typical usecase: nbasis -> 1
            Qt = np.ones((1, nbasis), np.complex128)

            # Bytes
            nbytes = (nbasis + 1) * result.global_bytes() * N

            # Time
            dt = 0.0
            for it in range(N + Nwarmup):
                if it >= Nwarmup:
                    dt -= g.time()
                g.linear_combination(result, basis, Qt)
                if it >= Nwarmup:
                    dt += g.time()

            # Report
            GBPerSec = nbytes / dt / 1e9
            g.message(
                f"""{N} linear_combination
    Object type                 : {tp.__name__}
    Number of basis vectors     : {nbasis}
    Time to complete            : {dt:.2f} s
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s
"""
            )


# main test loop
for precision in [g.single, g.double]:
    grid = g.grid(g.default.get_ivec("--grid", [16, 16, 16, 32], 4), precision)
    N = 100
    Nwarmup = 5
    g.message(
        f"""
Benchmark bilinear combination
    fdimensions  : {grid.fdimensions}
    precision    : {precision.__name__}
"""
    )

    # Source and destination
    for tp in [g.ot_singlet]:
        for nbasis in [4, 8, 16]:
            basis = [g.lattice(grid, tp) for i in range(nbasis)]
            result = [g.lattice(grid, tp) for i in range(nbasis)]
            rng.cnormal(basis)

            # Typical usecase : nbasis x nbasis -> nbasis
            Qt = np.ones((nbasis, nbasis), np.complex128)
            lidx = np.mod(np.arange(nbasis * nbasis, dtype=np.int32), nbasis).reshape(
                nbasis, nbasis
            )

            # Bytes
            nbytes = nbasis * (2 * nbasis + 1) * result[0].global_bytes() * N

            # Time
            dt = 0.0
            for it in range(N + Nwarmup):
                if it >= Nwarmup:
                    dt -= g.time()
                g.bilinear_combination(result, basis, basis, Qt, lidx, lidx)
                if it >= Nwarmup:
                    dt += g.time()

            # Report
            GBPerSec = nbytes / dt / 1e9
            g.message(
                f"""{N} bilinear_combination
    Object type                 : {tp.__name__}
    Number of basis vectors     : {nbasis}
    Time to complete            : {dt:.2f} s
    Effective memory bandwidth  : {GBPerSec:.2f} GB/s
"""
            )
