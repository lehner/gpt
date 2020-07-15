#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Check correctness of chiral splitting
#
import gpt as g
import numpy as np

# define grids
grid = g.grid([8, 8, 8, 8], g.double)

# setup rng
rng = g.random("ducks_smell_funny")

# size of basis
nbasis = 40
assert nbasis % 2 == 0
nb = nbasis // 2

# setup fine basis
basis_ref_f = [g.vspincolor(grid) for __ in range(nb)]
basis_split_f = [g.vspincolor(grid) for __ in range(nbasis)]
rng.cnormal(basis_ref_f)

# setup coarse basis
basis_ref_c = [g.vcomplex(grid, nbasis) for __ in range(nb)]
basis_split_c = [g.vcomplex(grid, nbasis) for __ in range(nbasis)]
rng.cnormal(basis_ref_c)


def run_test(basis_split, basis_ref):
    for factor in [0.5, 1.0, None]:
        for i in range(nb):
            basis_split[i] = g.copy(basis_ref[i])

        g.split_chiral(basis_split, factor)
        g.unsplit_chiral(basis_split, factor)

        typename = basis_split[0].otype.__name__
        for i in range(nb):
            diff2 = g.norm2(basis_ref[i] - basis_split[i])
            assert diff2 == 0.0
            if factor is None:
                g.message(
                    "Test passed (factor None, type %s) for vector %d, %e == 0.0"
                    % (typename, i, diff2)
                )
            else:
                g.message(
                    "Test passed (factor %g, type %s) for vector %d, %e == 0.0"
                    % (factor, typename, i, diff2)
                )


run_test(basis_split_f, basis_ref_f)
g.message("All tests for fine grid")
run_test(basis_split_c, basis_ref_c)
g.message("All tests for coarse grid")
