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

class modes:

    #
    # approximate A = sum_n left[n] right[n]^dag f(eval[n])
    #

    def __init__(self, left, right, evals, f):
        self.f = f
        self.left = left
        self.right = right
        self.evals = evals
        assert(len(left) == len(right) and
               len(left) == len(evals) and
               len(left) > 0)

    def __call__(self, matrix = None):

        # ignore matrix

        left = self.left
        right = self.right
        evals = self.evals
        f_evals = [ self.f(x) for x in evals ]

        otype = (left[0].otype,right[0].otype)
        grid = (left[0].grid,right[0].grid)
        cb = (left[0].checkerboard(),right[0].checkerboard())

        def approx(dst, src):
            assert(src != dst)
            verbose=g.default.is_verbose("modes")
            t0=g.time()
            dst[:]=0
            for i,x in enumerate(left):
                dst += f_evals[i] * x * g.innerProduct(right[i],src)
            if verbose:
                t1=g.time()
                g.message("Approximation by %d modes took %g s" % (len(left),t1-t0))
        
        return g.matrix_operator(mat = approx,
                                 otype = otype, zero = (False,False),
                                 grid = grid, cb = cb)
