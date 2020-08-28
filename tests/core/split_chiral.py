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
nbasis_f = 30
nbasis_c = 40
nb_f = nbasis_f // 2
nb_c = nbasis_c // 2

# setup fine basis
basis_ref_f = [g.vspincolor(grid) for __ in range(nb_f)]
basis_split_f = [g.vspincolor(grid) for __ in range(nbasis_f)]
rng.cnormal(basis_ref_f)

# setup coarse basis
basis_ref_c = [g.vcomplex(grid, nbasis_f) for __ in range(nb_c)]
basis_split_c = [g.vcomplex(grid, nbasis_f) for __ in range(nbasis_c)]
rng.cnormal(basis_ref_c)


def run_test(basis_split, basis_ref):
    for factor in [0.5, 1.0, None]:
        for i in range(len(basis_ref)):
            basis_split[i] = g.copy(basis_ref[i])

        g.coarse.split_chiral(basis_split, factor)
        g.coarse.unsplit_chiral(basis_split, factor)

        typename = basis_split[0].otype.__name__
        for i in range(len(basis_ref)):
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
