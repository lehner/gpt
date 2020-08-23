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
# Note: we use the proper order of the chebyshev_t
#       in contrast to current Grid
#
import gpt as g
import numpy as np
from gpt.params import params_convention


class deflate:
    @params_convention(block=16)
    def __init__(self, inverter, evec, ev, params):
        self.inverter = inverter
        self.evec = evec
        self.ev = ev
        self.params = params

    def __call__(self, matrix):

        otype = self.evec[0].otype
        grid = self.evec[0].grid
        cb = self.evec[0].checkerboard()

        inverter = self.inverter(matrix)

        def inv(dst, src):
            verbose = g.default.is_verbose("deflate")
            # |dst> = sum_n 1/ev[n] |n><n|src>
            t0 = g.time()
            grid = src[0].grid
            rip = np.zeros((len(src), len(self.evec)), dtype=np.complex128)
            block = self.params["block"]
            for i0 in range(0, len(self.evec), block):
                rip_block = g.rank_inner_product(self.evec[i0 : i0 + block], src, True)
                for i in range(rip_block.shape[0]):
                    for j in range(rip_block.shape[1]):
                        rip[j, i0 + i] = rip_block[i, j] / self.ev[i0 + i]
            t1 = g.time()
            grid.globalsum(rip)
            t2 = g.time()
            # TODO: simultaneous linear_combinations
            for j in range(len(src)):
                g.linear_combination(dst[j], self.evec, rip[j])
            t3 = g.time()
            if verbose:
                g.message(
                    "Deflated %d vector(s) in %g s (%g s for rank_inner_product, %g s for global sum, %g s for linear combinations)"
                    % (len(src), t3 - t0, t1 - t0, t2 - t1, t3 - t2)
                )
            return inverter(dst, src)

        return g.matrix_operator(
            mat=inv, inv_mat=matrix, otype=otype, grid=grid, cb=cb, accept_list=True
        )
