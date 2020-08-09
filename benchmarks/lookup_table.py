#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

# helper function
def assert_relative_deviation(reference, result, tol):
    diff = g.eval(reference - result)
    abs_dev = g.norm2(diff)
    rel_dev = abs_dev / g.norm2(reference)
    g.message(f"abs. dev. = {abs_dev}, rel. dev. = {rel_dev}")
    assert rel_dev <= tol


# command line parameters
niter = g.default.get_int("--niter", 100)
nbasis = g.default.get_int("--nbasis", 40)
fgrid_size = g.default.get_ivec("--fgrid", [8, 8, 8, 8], 4)
cgrid_size = g.default.get_ivec("--cgrid", [4, 4, 4, 4], 4)

# setup grids
fine_grid = g.grid(fgrid_size, g.single)
coarse_grid = g.grid(cgrid_size, fine_grid.precision)

# setup rng
rng = g.random("ducks_smell_funny")

# create masks
full_mask, blockeven_mask = g.complex(fine_grid), g.complex(fine_grid)
full_mask[:] = 1
coor = g.coordinates(blockeven_mask)
block = np.array(fine_grid.ldimensions) / np.array(coarse_grid.ldimensions)
block_cb = coor[:, :] // block[:]
g.make_mask(blockeven_mask, np.sum(block_cb, axis=1) % 2 == 0)

# create luts
full_lut = g.lookup_table(coarse_grid, full_mask)
blockeven_lut = g.lookup_table(coarse_grid, blockeven_mask)

# setup basis
basis = [g.vspincolor(fine_grid) for __ in range(nbasis)]
rng.cnormal(basis)

# setup fields
fvec_in = g.vspincolor(fine_grid)
cvec_out_default = g.vcomplex(coarse_grid, nbasis)
cvec_out_mask = g.vcomplex(coarse_grid, nbasis)
cvec_out_lut = g.vcomplex(coarse_grid, nbasis)
cscal_out_tmp = g.complex(coarse_grid)
cmat_out_mask = g.mcomplex(coarse_grid, nbasis)
cmat_out_lut = g.mcomplex(coarse_grid, nbasis)

# initialize fields
rng.cnormal(fvec_in)
cvec_out_default[:] = 0.0
cvec_out_mask[:] = 0.0
cvec_out_lut[:] = 0.0
cscal_out_tmp[:] = 0.0
cmat_out_mask[:] = 0.0
cmat_out_lut[:] = 0.0

# define check tolerance
tol = 1e-15

# setup timers
td_ref, td_new = 0.0, 0.0

# run normal block project (full mask)
td_ref -= g.time()
for n in range(niter):
    g.block.project(cvec_out_default, fvec_in, basis)
td_ref += g.time()
g.message(f"block.project took {td_ref} s for {niter} iter")

# run block project with lut (full mask)
td_new -= g.time()
for n in range(niter):
    g.block.project_using_lut(cvec_out_lut, fvec_in, basis, full_lut)
td_new += g.time()
g.message(f"block.project_using_lut took {td_new} s for {niter} iter")

# report speedup
g.message(f"Speedup block.project_using_lut vs block.project is {td_ref/td_new} x")

# report error
assert_relative_deviation(cvec_out_default, cvec_out_lut, tol)

# setup timers
td_ref, td_new = 0.0, 0.0
t0, t1, t2 = 0.0, 0.0, 0.0

# run block masked inner product (mask with zeros)
td_ref -= g.time()
for n in range(niter):
    for j, bv in enumerate(basis):
        t0 += g.time()
        g.block.masked_inner_product(cscal_out_tmp, blockeven_mask, bv, fvec_in)
        t1 += g.time()
        g.block.matrix_elem_from_scalar(cmat_out_mask, [j, 0], cscal_out_tmp)
        t2 += g.time()
td_ref += g.time()
g.message(
    f"block.masked_inner_product took {td_ref} s for {niter} iter ({t1-t0} s work, {t2-t1} s copy)"
)

# setup timers
t0, t1, t2 = 0.0, 0.0, 0.0

# run block project product using lut (lut with zeros)
td_new -= g.time()
for n in range(niter):
    t0 += g.time()
    g.block.project_using_lut(cvec_out_lut, fvec_in, basis, blockeven_lut)
    t1 += g.time()
    g.block.matrix_column_from_vector(cmat_out_lut, 0, cvec_out_lut)
    t2 += g.time()
td_new += g.time()
g.message(
    f"block.project_using_lut took {td_new} s for {niter} iter ({t1-t0} s work, {t2-t1} s copy)"
)

# report speedup
g.message(
    f"Speedup block.project_using_lut vs block.masked_inner_product is {td_ref/td_new} x"
)

# report error
assert_relative_deviation(cmat_out_mask, cmat_out_lut, tol)
