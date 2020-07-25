#!/usr/bin/env python3
#
# Authors: Tilo Wettig 2020
#          Christoph Lehner 2020
#
# test code for new gauge field types
#
import gpt as g
import numpy as np
import sys, cgpt

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
U = g.qcd.gauge.random(grid_sp, rng, otype=g.ot_matrix_su2_fundamental())
eps = abs(g.qcd.gauge.plaquette(U) - 0.8813162545363108)
g.message("Test SU(2) fundamental single", eps)
assert eps < 1e-7
check_unitarity(U, 1e-7)

V = g.qcd.gauge.fundamental_to_adjoint(U)
eps = abs(g.qcd.gauge.plaquette(V) - 0.7126823001437717)
g.message("Test SU(2) fundamental to adjoint single", eps)
assert eps < 1e-7
check_unitarity(V, 1e-7)

rng = g.random("test")
U = g.qcd.gauge.random(grid_dp, rng, otype=g.ot_matrix_su2_fundamental())
eps = abs(g.qcd.gauge.plaquette(U) - 0.8813162591343201)
g.message("Test SU(2) fundamental double", eps)
check_unitarity(U, 1e-14)

assert eps < 1e-14
V = g.qcd.gauge.fundamental_to_adjoint(U)
eps = abs(g.qcd.gauge.plaquette(V) - 0.7126822868786024)
g.message("Test SU(2) fundamental to adjoint double", eps)
assert eps < 1e-14
check_unitarity(V, 1e-14)

################################################################################
# Test SU(2) adjoint
################################################################################

rng = g.random("test")
U = g.qcd.gauge.random(grid_sp, rng, otype=g.ot_matrix_su2_adjoint())
eps = abs(g.qcd.gauge.plaquette(U) - 0.712682286898295)
g.message("Test SU(2) adjoint single", eps)
assert eps < 1e-7
check_unitarity(U, 1e-7)

rng = g.random("test")
U = g.qcd.gauge.random(grid_dp, rng, otype=g.ot_matrix_su2_adjoint())
eps = abs(g.qcd.gauge.plaquette(U) - 0.7126822868786024)
g.message("Test SU(2) adjoint double", eps)
assert eps < 1e-14
check_unitarity(U, 1e-14)

