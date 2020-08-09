#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test constructed coarse and coarse coarse operator against naive chained application
#
import gpt as g
import numpy as np
import sys

# command line parameters
grid_f_size = g.default.get_ivec("--fgrid", [8, 8, 8, 8], 4)
grid_c_size = g.default.get_ivec("--cgrid", [4, 4, 4, 4], 4)
grid_cc_size = g.default.get_ivec("--ccgrid", [2, 2, 2, 2], 4)

# setup fine link fields
U = g.qcd.gauge.random(g.grid(grid_f_size, g.double), g.random("test"))

# do everything in single precision
U = g.convert(U, g.single)

# setup grids
grid_f = U[0].grid
grid_c = g.grid(grid_c_size, grid_f.precision)
grid_cc = g.grid(grid_cc_size, grid_f.precision)

# setup fine matrix
mat_f = g.qcd.fermion.wilson_clover(
    U,
    {
        "mass": -0.1,
        "csw_r": 0,
        "csw_t": 0,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)

# setup rng
rng = g.random("ducks_smell_funny")

# number of basis vectors
nbasis_f = 30
nbasis_c = 40

# number of block orthogonalization steps
northo = 2

# define check tolerances
tol_ortho = 1e-28 if grid_f.precision == g.double else 1e-11
tol_links = 1e-30 if grid_f.precision == g.double else 1e-13
tol_operator = 1e-30 if grid_f.precision == g.double else 1e-13

# setup fine basis
basis_f = [g.vspincolor(grid_f) for __ in range(nbasis_f)]
rng.cnormal(basis_f)

# split fine basis into chiral halfs
g.split_chiral(basis_f)

# orthonormalize fine basis
for i in range(northo):
    g.message("Block ortho step %d" % i)
    g.block.orthonormalize(grid_c, basis_f)

# check orthogonality
g.block.check_orthogonality(grid_c, basis_f, tol_ortho)
g.message("Orthogonality check for fine basis done")

# create coarse link fields
A_c = [g.mcomplex(grid_c, nbasis_f) for __ in range(9)]
A_lut_c = [g.mcomplex(grid_c, nbasis_f) for __ in range(9)]
Asaved_c = [g.mcomplex(grid_c, nbasis_f) for __ in range(9)]
Asaved_lut_c = [g.mcomplex(grid_c, nbasis_f) for __ in range(9)]
g.coarse.create_links(
    A_c, mat_f, basis_f, {"hermitian": False, "savelinks": False, "uselut": False}
)
g.coarse.create_links(
    A_lut_c, mat_f, basis_f, {"hermitian": False, "savelinks": False, "uselut": True}
)
g.coarse.create_links(
    Asaved_c, mat_f, basis_f, {"hermitian": False, "savelinks": True, "uselut": False}
)
g.coarse.create_links(
    Asaved_lut_c,
    mat_f,
    basis_f,
    {"hermitian": False, "savelinks": True, "uselut": True},
)

# compare link fields
for p in range(9):
    err2 = g.norm2(A_c[p] - A_lut_c[p]) / g.norm2(A_c[p])
    g.message(f"Relative deviation of A_lut_c[{p}] from A_c[{p}] = {err2:e}",)
    assert err2 <= tol_links
    err2 = g.norm2(A_c[p] - Asaved_c[p]) / g.norm2(A_c[p])
    g.message(f"Relative deviation of Asaved_c[{p}] from A_c[{p}] = {err2:e}",)
    assert err2 <= tol_links
    err2 = g.norm2(A_c[p] - Asaved_lut_c[p]) / g.norm2(A_c[p])
    g.message(f"Relative deviation of Asaved_lut_c[{p}] from A_c[{p}] = {err2:e}",)
    assert err2 <= tol_links
g.message(f"Tests for links passed for all directions")
del A_lut_c, Asaved_c, Asaved_lut_c

# create coarse operator from links
mat_c = g.qcd.fermion.coarse(A_c, {"level": 0,},)

# setup fine vectors
vec_in_f = g.lattice(basis_f[0])
vec_out_f = g.lattice(basis_f[0])
vec_in_f[:] = 0.0
vec_out_f[:] = 0.0

# setup coarse vectors
vec_in_c = g.vcomplex(grid_c, nbasis_f)
vec_out_chained_c = g.vcomplex(grid_c, nbasis_f)
vec_out_constructed_c = g.vcomplex(grid_c, nbasis_f)
rng.cnormal(vec_in_c)
vec_out_chained_c[:] = 0.0
vec_out_constructed_c[:] = 0.0

# apply chained and constructed coarse operator
dt_chained, dt_constructed = 0.0, 0.0
dt_chained -= g.time()
g.block.promote(vec_in_c, vec_in_f, basis_f)
mat_f(vec_out_f, vec_in_f)
g.block.project(vec_out_chained_c, vec_out_f, basis_f)
dt_chained += g.time()
dt_constructed -= g.time()
mat_c(vec_out_constructed_c, vec_in_c)
dt_constructed += g.time()

g.message("Timings: chained = %e, constructed = %e" % (dt_chained, dt_constructed))

# report error
err2 = g.norm2(vec_out_chained_c - vec_out_constructed_c) / g.norm2(vec_out_chained_c)
g.message(
    "Relative deviation of constructed from chained coarse operator on coarse grid = %e"
    % err2
)
assert err2 <= tol_operator
g.message("Test passed for coarse operator, %e <= %e" % (err2, tol_operator))

# Done with fine grid, now test on coarse #####################################

# setup coarse basis
basis_c = [g.vcomplex(grid_c, nbasis_f) for __ in range(nbasis_c)]
rng.cnormal(basis_c)

# split coarse basis into chiral halfs
g.split_chiral(basis_c)

# orthonormalize coarse basis
for i in range(northo):
    g.message("Block ortho step %d" % i)
    g.block.orthonormalize(grid_cc, basis_c)

# check orthogonality
g.block.check_orthogonality(grid_cc, basis_c, tol_ortho)
g.message("Orthogonality check for coarse basis done")

# create coarse coarse link fields
A_cc = [g.mcomplex(grid_cc, nbasis_c) for __ in range(9)]
A_lut_cc = [g.mcomplex(grid_cc, nbasis_c) for __ in range(9)]
Asaved_cc = [g.mcomplex(grid_cc, nbasis_c) for __ in range(9)]
Asaved_lut_cc = [g.mcomplex(grid_cc, nbasis_c) for __ in range(9)]
g.coarse.create_links(
    A_cc, mat_c, basis_c, {"hermitian": False, "savelinks": False, "uselut": False}
)
g.coarse.create_links(
    A_lut_cc, mat_c, basis_c, {"hermitian": False, "savelinks": False, "uselut": True}
)
g.coarse.create_links(
    Asaved_cc, mat_c, basis_c, {"hermitian": False, "savelinks": True, "uselut": False}
)
g.coarse.create_links(
    Asaved_lut_cc,
    mat_c,
    basis_c,
    {"hermitian": False, "savelinks": True, "uselut": True},
)

# compare link fields
for p in range(9):
    err2 = g.norm2(A_cc[p] - A_lut_cc[p]) / g.norm2(A_cc[p])
    g.message(f"Relative deviation of A_lut_cc[{p}] from A_cc[{p}] = {err2:e}",)
    assert err2 <= tol_links
    err2 = g.norm2(A_cc[p] - Asaved_cc[p]) / g.norm2(A_cc[p])
    g.message(f"Relative deviation of Asaved_cc[{p}] from A_cc[{p}] = {err2:e}",)
    assert err2 <= tol_links
    err2 = g.norm2(A_cc[p] - Asaved_lut_cc[p]) / g.norm2(A_cc[p])
    g.message(f"Relative deviation of Asaved_lut_cc[{p}] from A_cc[{p}] = {err2:e}",)
    assert err2 <= tol_links
g.message(f"Tests for links passed for all directions")
del A_lut_cc, Asaved_cc, Asaved_lut_cc

# create coarse operator from links
mat_cc = g.qcd.fermion.coarse(A_cc, {"level": 1,},)

# setup coarse vectors
vec_out_c = g.lattice(basis_c[0])
vec_in_c[:] = 0.0
vec_out_c[:] = 0.0

# setup coarse coarse vectors
vec_in_cc = g.vcomplex(grid_cc, nbasis_c)
vec_out_chained_cc = g.vcomplex(grid_cc, nbasis_c)
vec_out_constructed_cc = g.vcomplex(grid_cc, nbasis_c)
rng.cnormal(vec_in_cc)
vec_out_chained_cc[:] = 0.0
vec_out_constructed_cc[:] = 0.0

# apply chained and constructed coarse coarse operator
dt_chained, dt_constructed = 0.0, 0.0
dt_chained -= g.time()
g.block.promote(vec_in_cc, vec_in_c, basis_c)
mat_c(vec_out_c, vec_in_c)
g.block.project(vec_out_chained_cc, vec_out_c, basis_c)
dt_chained += g.time()
dt_constructed -= g.time()
mat_cc(vec_out_constructed_cc, vec_in_cc)
dt_constructed += g.time()

g.message("Timings: chained = %e, constructed = %e" % (dt_chained, dt_constructed))

# report error
err2 = g.norm2(vec_out_chained_cc - vec_out_constructed_cc) / g.norm2(
    vec_out_chained_cc
)
g.message(
    "Relative deviation of constructed from chained coarse coarse operator on coarse coarse grid = %e"
    % err2
)
assert err2 <= tol_operator
g.message("Test passed for coarse coarse operator, %e <= %e" % (err2, tol_operator))

# Done with chaining tests, now test ET against Grid on coarse grid ###########

# setup fields
rng.cnormal(A_c)
mat_c = g.qcd.fermion.coarse(A_c, {"level": 0,},)
vec_out_link_c, vec_out_mat_c = g.lattice(vec_in_c), g.lattice(vec_in_c)
vec_out_link_c[:] = 0.0
vec_out_link_c[:] = 0.0
rng.cnormal(vec_in_c)

# apply the link matrix
vec_out_link_c @= A_c[8] * vec_in_c
mat_c.Mdir(
    vec_out_mat_c, vec_in_c, 0, 0
)  # exploit the self coupling link, this uses Grid

# define check tolerance
tol = 0.0

# report error
diff2 = g.norm2(vec_out_link_c - vec_out_mat_c)
assert diff2 == tol
g.message("Test passed for coarse links, %e == %e" % (diff2, tol))

# Test ET against Grid on coarse coarse grid ##################################

# setup fields
rng.cnormal(A_cc)
mat_cc = g.qcd.fermion.coarse(A_cc, {"level": 1,},)
vec_out_link_cc, vec_out_mat_cc = g.lattice(vec_in_cc), g.lattice(vec_in_cc)
vec_out_link_cc[:] = 0.0
vec_out_link_cc[:] = 0.0
rng.cnormal(vec_in_cc)

# apply the link matrix
vec_out_link_cc @= A_cc[8] * vec_in_cc
mat_cc.Mdir(
    vec_out_mat_cc, vec_in_cc, 0, 0
)  # exploit the self coupling link, this uses Grid

# define check tolerance
tol = 0.0

# report error
diff2 = g.norm2(vec_out_link_cc - vec_out_mat_cc)
assert diff2 == tol
g.message("Test passed for coarse coarse links, %e == %e" % (diff2, tol))
