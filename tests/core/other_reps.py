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
  eps = ( g.norm2(U * g.adj(U) - eye) / g.norm2(eye) ) ** 0.5
  g.message(f"Test unitarity: {eps}")
  assert eps < eps_ref

################################################################################
# Test SU(2) fundamental and conversion to adjoint
################################################################################

rng = g.random("test")

U = g.matrix_su2_fundamental(grid_sp)
rng.lie(U)
check_unitarity(U, 1e-7)

# TODO: also make fundamental_to_adjoint a matrix_operator
V = g.qcd.gauge.fundamental_to_adjoint(U)
check_unitarity(V, 1e-7)

U = g.matrix_su2_fundamental(grid_dp)
rng.lie(U)
check_unitarity(U, 1e-14)

V = g.qcd.gauge.fundamental_to_adjoint(U)
check_unitarity(V, 1e-14)

# TODO: the tests should also check the irrep


################################################################################
# Test SU(2) adjoint
################################################################################

rng = g.random("test")
U = g.matrix_su2_adjoint(grid_sp)
rng.lie(U)
check_unitarity(U, 1e-7)

rng = g.random("test")
U = g.matrix_su2_adjoint(grid_dp)
rng.lie(U)
check_unitarity(U, 1e-14)
