#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test constructed coarse operator against naive chained application
#
import gpt as g
import numpy as np
import sys

# setup fine link fields
U = g.qcd.gauge.random(g.grid([8, 8, 8, 8], g.double), g.random("test"))

# do everything in single precision
U = g.convert(U, g.single)

# setup grids
fine_grid = U[0].grid
coarse_grid = g.grid([2, 2, 2, 2], fine_grid.precision)

# setup fine matrix
fmat = g.qcd.fermion.wilson_clover(U, {
    "kappa" : 0.13565,
    "csw_r" : 2.0171,
    "csw_t" : 2.0171,
    "xi_0" : 1,
    "nu" : 1,
    "isAnisotropic" : False,
    "boundary_phases" : [ 1.0, 1.0, 1.0, 1.0 ]
})

# setup rng
rng = g.random("ducks_smell_funny")

# number of basis vectors
nbasis = 20

# number of block orthogonalization steps
northo = 2

# setup basis
basis = [ g.vspincolor(fine_grid) for i in range(nbasis) ]
rng.cnormal(basis)
g.split_chiral(basis)

# orthonormalize basis
for i in range(northo):
    g.message("Block ortho step %d" % i)
    g.block.orthonormalize(coarse_grid, basis)

# check orthogonality
iproj, eproj = g.vcomplex(coarse_grid, nbasis), g.vcomplex(coarse_grid, nbasis)
for i, v in enumerate(basis):
    g.block.project(iproj, v, basis)
    eproj[:] = 0.0
    eproj[:, :, :, :, i] = 1.0
    err2 = g.norm2(eproj - iproj)
    g.message("Orthogonality check error for vector %d = %e" % (i, err2))
g.message("Orthogonality check done")

# create coarse link fields
A = [g.mcomplex(coarse_grid, nbasis) for i in range(9)]
g.coarse.create_links(A, fmat, basis)

# create coarse operator from links
cmat = g.qcd.fermion.coarse_operator(A, {
    'hermitian': 1,
    'level': 0,
    'nbasis': nbasis, # get the size from clinks and ditch the parameter?
})

# setup fine vectors
fvec_in = g.lattice(basis[0])
fvec_out = g.lattice(basis[0])
fvec_in[:] = 0
fvec_out[:] = 0

# setup coarse vectors
cvec_in = g.vcomplex(coarse_grid, nbasis)
cvec_out_chained = g.vcomplex(coarse_grid, nbasis)
cvec_out_constructed = g.vcomplex(coarse_grid, nbasis)
rng.cnormal(cvec_in)
cvec_out_chained[:] = 0
cvec_out_constructed[:] = 0

# apply chained and constructed coarse operator
dt_chained, dt_constructed = 0.0, 0.0
dt_chained -= g.time()
g.block.promote(cvec_in, fvec_in, basis)
fmat.M(fvec_out, fvec_in)
g.block.project(cvec_out_chained, fvec_out, basis)
dt_chained += g.time()
dt_constructed -= g.time()
cmat.M(cvec_out_constructed, cvec_in)
dt_constructed += g.time()

g.message("Timings: chained = %e, constructed = %e" % (dt_chained, dt_constructed))

# define check tolerance
tol = 1e-26 if fine_grid.precision == g.double else 1e-13

# report error
err2 = g.norm2(cvec_out_chained - cvec_out_constructed) / g.norm2(cvec_out_chained)
g.message("Relative deviation of constructed from chained operator = %e" % err2)
assert err2 <= tol
g.message("Test passed, %e <= %e" % (err2, tol))
