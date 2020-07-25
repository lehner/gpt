#!/usr/bin/env python3
#
# Authors: Tilo Wettig 2020
#          Christoph Lehner 2020
#
# test code for new gauge field types
#
import gpt as g
import numpy as np

# grid
L = [8, 4, 4, 4]
grid_dp = g.grid(L, g.double)
grid_sp = g.grid(L, g.single)

# unitarity test
def check_unitarity(U, eps_ref):
    eye = g.lattice(U)
    eye[:] = np.eye(U.otype.shape[0], dtype=U.grid.precision.complex_dtype)
    eps = (g.norm2(U * g.adj(U) - eye) / g.norm2(eye)) ** 0.5
    g.message(f"Test unitarity: {eps}")
    assert eps < eps_ref


def check_representation(U, eps_ref):
    generators = U.otype.generators(U.grid.precision.complex_dtype)

    # first test generators normalization
    for a in range(len(generators)):
        for b in range(len(generators)):
            eye_ab = 2.0 * g.trace(generators[a] * generators[b])
            if a == b:
                assert abs(eye_ab - 1) < eps_ref
            else:
                assert abs(eye_ab) < eps_ref

    # now project to algebra and make sure it is a linear combination of
    # the provided generators
    algebra = g.matrix.log(U)
    algebra /= 1j
    n0 = g.norm2(algebra)
    for Ta in generators:
        algebra -= Ta * g.trace(algebra * Ta) * 2.0
    eps = (g.norm2(algebra) / n0) ** 0.5
    g.message(f"Test representation: {eps}")
    assert eps < eps_ref


################################################################################
# Test SU(2) fundamental and conversion to adjoint
################################################################################

rng = g.random("test")

for eps_ref, grid in [(1e-6, grid_sp), (1e-12, grid_dp)]:
    g.message(
        f"Test SU(2) fundamental and adjoint conversion on grid {grid.precision.__name__}"
    )

    U = g.matrix_su2_fundamental(grid)
    rng.lie(U)
    check_unitarity(U, eps_ref)
    check_representation(U, eps_ref)

    V = g.matrix_su2_adjoint(grid)
    g.qcd.gauge.fundamental_to_adjoint(V, U)
    check_unitarity(V, eps_ref)
    check_representation(V, eps_ref)


################################################################################
# Test all other representations
################################################################################

for eps_ref, grid in [(1e-7, grid_sp), (1e-14, grid_dp)]:
    for representation in [g.matrix_su2_adjoint, g.matrix_su3_fundamental]:
        g.message(f"Test {representation.__name__} on grid {grid.precision.__name__}")
        U = representation(grid)
        rng.lie(U)
        check_unitarity(U, eps_ref)
        check_representation(U, eps_ref)
