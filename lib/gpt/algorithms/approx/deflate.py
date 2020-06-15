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

class deflate:
    def __init__(self, inverter, evec, ev):
        self.inverter = inverter
        self.evec = evec
        self.ev = ev

    def __call__(self, matrix):

        otype = None
        grid = None
        if type(matrix) == g.matrix_operator:
            otype = matrix.otype
            grid = matrix.grid
            matrix = matrix.mat

        def inv(dst, src):
            verbose=g.default.is_verbose("deflate")
            # |dst> = sum_n 1/ev[n] |n><n|src>
            t0=g.time()
            dst[:]=0
            for i,n in enumerate(self.evec):
                dst += n*g.innerProduct(n,src)/self.ev[i]
            t1=g.time()
            if verbose:
                g.message("Deflated in %g s" % (t1-t0))
            return self.inverter(matrix)(dst, src)

        return g.matrix_operator(mat = inv, inv_mat = matrix, otype = otype)
