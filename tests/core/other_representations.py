#!/usr/bin/env python3
#
# Authors: Tilo Wettig 2020
#          Christoph Lehner 2020
#
# test code for new gauge field types
#
import gpt as g
import numpy as np
import sys

# grid
L = [8, 4, 4, 4]
grid_dp = g.grid(L, g.double)
grid_sp = g.grid(L, g.single)
rng = g.random("test")

# unitarity test
def check_unitarity(U, eps_ref):
    eye = g.lattice(U)
    eye[:] = np.eye(U.otype.shape[0], dtype=U.grid.precision.complex_dtype)
    eps = (g.norm2(U * g.adj(U) - eye) / g.norm2(eye)) ** 0.5
    g.message(f"Test unitarity: {eps}")
    assert eps < eps_ref
    U.otype.is_element(U)


def check_representation(U, eps_ref):
    algebra = g.convert(U, U.otype.cartesian())

    # then test coordinates function
    algebra2 = g.lattice(algebra)
    algebra2[:] = 0
    algebra2.otype.coordinates(algebra2, algebra.otype.coordinates(algebra))
    eps = (g.norm2(algebra2 - algebra) / g.norm2(algebra)) ** 0.5
    g.message(f"Test coordinates: {eps}")
    assert eps < eps_ref

    # now project to algebra and make sure it is a linear combination of
    # the provided generators
    n0 = g.norm2(algebra)
    algebra2.otype.coordinates(
        algebra2, g.component.real(algebra.otype.coordinates(algebra))
    )
    algebra -= algebra2
    eps = (g.norm2(algebra) / n0) ** 0.5
    g.message(f"Test representation: {eps}")
    assert eps < eps_ref


################################################################################
# Test projection schemes on promoting SP to DP group membership
################################################################################
V0 = g.convert(rng.element(g.mcolor(grid_sp)), g.double)
for method in ["defect_left", "defect_right"]:
    V = g.copy(V0)
    I = g.identity(V)
    I_s = g.identity(g.complex(grid_dp))
    for i in range(3):
        eps_uni = (g.norm2(g.adj(V) * V - I) / g.norm2(I)) ** 0.5
        eps_det = (g.norm2(g.matrix.det(V) - I_s) / g.norm2(I_s)) ** 0.5
        g.message(
            f"Before {method} iteration {i}, unitarity defect: {eps_uni}, determinant defect: {eps_det}"
        )
        g.project(V, method)
    assert eps_uni < 1e-14 and eps_det < 1e-14


################################################################################
# Test SU(2) fundamental and conversion to adjoint
################################################################################

rng = g.random("test")

for eps_ref, grid in [(1e-6, grid_sp), (1e-12, grid_dp)]:
    g.message(
        f"Test SU(2) fundamental and adjoint conversion on grid {grid.precision.__name__}"
    )

    U = [g.matrix_su2_fundamental(grid) for i in range(2)]
    rng.element(
        U, scale=0.2
    )  # need to stay close to identity element for mappings to be unique
    U.append(g.eval(U[0] * U[1]))

    for u in U:
        check_unitarity(u, eps_ref)
        check_representation(u, eps_ref)

    V = [g.matrix_su2_adjoint(grid) for x in U]
    for i in range(3):
        g.convert(
            V[i], U[i]
        )  # this used to be a separate function: fundamental_to_adjoint
        check_unitarity(V[i], eps_ref)
        check_representation(V[i], eps_ref)

    # check if fundamental_to_adjoint is a homomorphism
    eps = (g.norm2(V[2] - V[0] * V[1]) / g.norm2(V[2])) ** 0.5
    g.message(f"Test fundamental_to_adjoint is homomorphism: {eps}")
    assert eps < eps_ref

    a = [
        g.lattice(V[0].grid, g.ot_matrix_su_n_fundamental_algebra(2)) for i in range(3)
    ]
    g.convert(a, U)

    V_c = []
    for i in range(3):
        # convert through canonical coordinates
        coor = a[i].otype.coordinates(a[i])
        a_adj = g.lattice(a[i].grid, g.ot_matrix_su_n_adjoint_algebra(2))
        a_adj.otype.coordinates(a_adj, coor)
        v = g.convert(a_adj, g.ot_matrix_su_n_adjoint_group(2))
        check_unitarity(v, eps_ref)
        check_representation(v, eps_ref)
        V_c.append(v)

    # check if coordinate transformation is a homomorphism
    eps = (g.norm2(V_c[2] - V_c[0] * V_c[1]) / g.norm2(V_c[2])) ** 0.5
    g.message(f"Test coordinate transformation is homomorphism: {eps}")
    assert eps < eps_ref

    # check identity of coordinate transformation and direct fundamental to adjoint
    for i in range(3):
        eps = (g.norm2(V_c[i] - V[i]) / g.norm2(V[i])) ** 0.5
        g.message(
            f"Identity of coordinate transformation and fundamental to adjoint: {eps}"
        )
        assert eps < eps_ref


################################################################################
# Test all other representations
################################################################################
for eps_ref, grid in [(1e-6, grid_sp), (1e-12, grid_dp)]:
    for representation in [g.matrix_su2_adjoint, g.matrix_su3_fundamental]:
        g.message(f"Test {representation.__name__} on grid {grid.precision.__name__}")
        U = representation(grid)
        rng.element(U)
        check_unitarity(U, eps_ref)
        check_representation(U, eps_ref)


################################################################################
# Test su2 subalgebras
################################################################################
for eps_ref, grid in [(1e-12, grid_dp)]:
    U = g.lattice(grid, g.ot_matrix_su_n_fundamental_group(3))
    u2 = g.lattice(grid, g.ot_matrix_su_n_fundamental_group(2))
    u2p = g.lattice(grid, g.ot_matrix_su_n_fundamental_group(2))
    for sg in U.otype.su2_subgroups():

        rng.element(u2)
        u2p = g.eval(u2 - g.adj(u2) + g.identity(u2) * g.trace(g.adj(u2)))
        eps = (g.norm2(u2 - u2p) / g.norm2(u2)) ** 0.5
        g.message(eps, g.norm2(u2), g.norm2(u2p))

        U.otype.block_insert(U, u2, sg)
        u2p[:] = 0
        U.otype.block_extract(u2p, U, sg)

        eps = (g.norm2(u2 - u2p) / g.norm2(u2)) ** 0.5
        g.message(eps, g.norm2(u2), g.norm2(u2p))
        assert eps < eps_ref

        check_unitarity(U, eps_ref)
        check_representation(U, eps_ref)
