#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import gpt as g
import numpy as np
from gpt.params import params_convention


#
# Create a matrix A from a mode representation
#
#  A = sum_n left[n] right[n]^dag f(eval[n])
#
@params_convention(block=16, linear_combination_block=8)
def matrix(left, right, evals, f, params):
    assert len(left) == len(right) and len(left) <= len(evals) and len(left) > 0
    f_evals = [f(x) for x in evals]

    def approx(dst, src):
        assert src != dst
        verbose = g.default.is_verbose("modes")

        t0 = g.time()
        grid = src[0].grid
        rip = np.zeros((len(src), len(left)), dtype=np.complex128)
        block = params["block"]
        linear_combination_block = params["linear_combination_block"]
        for i0 in range(0, len(left), block):
            rip_block = g.rank_inner_product(right[i0 : i0 + block], src, True)
            for i in range(rip_block.shape[0]):
                for j in range(rip_block.shape[1]):
                    rip[j, i0 + i] = rip_block[i, j] * f_evals[i0 + i]
        t1 = g.time()
        grid.globalsum(rip)
        t2 = g.time()
        g.linear_combination(dst, left, rip, linear_combination_block)
        t3 = g.time()
        if verbose:
            g.message(
                "Mode-representation applied to %d vector(s) in %g s (%g s for rank_inner_product, %g s for global sum, %g s for linear combinations)"
                % (len(src), t3 - t0, t1 - t0, t2 - t1, t3 - t2)
            )

    return g.matrix_operator(
        mat=approx,
        accept_guess=(False, False),
        vector_space=(
            g.vector_space.explicit_lattice(left[0]),
            g.vector_space.explicit_lattice(right[0]),
        ),
        accept_list=True,
    )
