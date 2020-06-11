#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test instantiation of coarse operator for a high number of levels
#
import gpt as g
import numpy as np
import sys

# NOTE: We should test composite against native here!

# setup grid
grid=g.grid([4,4,4,4],g.single)

# setup rng
rng=g.random("ducks_smell_funny")

# number of basis vectors
nbasis = 20

# number of mg levels to use
# 1 = only fine, 2 = 1 cooarse level, ...
mg_levels = 18

# coarse link fields
A = [g.mcomplex(grid, nbasis) for i in range(9)]

# setup coarse operators + vectors for the respective levels
num_coarse_levels = mg_levels - 1
cop = []
cvec_in = []
cvec_out = []
for clvl in range(num_coarse_levels):
    mglvl = clvl + 1
    cop.append(g.qcd.fermion.coarse_operator(A, {
        'hermitian': 1,
        'level': clvl,
        'nbasis': nbasis,
    }))
    cvec_in.append(g.vcomplex(grid, nbasis))
    cvec_out.append(g.vcomplex(grid, nbasis))
    rng.cnormal(cvec_in[clvl])
    cvec_out[clvl][:] = 0
    g.message("mglvl = %d, clvl = %d: Finished setup" % (mglvl, clvl))

# apply coarse operator
for clvl in range(num_coarse_levels):
    mglvl = clvl + 1
    cop[clvl].M(cvec_out[clvl], cvec_in[clvl])
    g.message("mglvl = %d, clvl = %d: in = %e" % (mglvl, clvl, g.norm2(cvec_in[clvl])))
    g.message("mglvl = %d, clvl = %d: out = %e" % (mglvl, clvl, g.norm2(cvec_out[clvl])))
