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

class coarse_deflate:
    def __init__(self,inverter, cevec, basis, fev):
        self.inverter = inverter
        self.cevec    = cevec
        self.fev      = fev
        self.basis    = basis
        self.csrc     = g.lattice(cevec[0])
        self.cdst     = g.lattice(cevec[0])

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
            g.block.project(self.csrc,src,self.basis)
            t1=g.time()
            self.cdst[:]=0
            for i,n in enumerate(self.cevec):
                self.cdst += n*g.innerProduct(n,self.csrc)/self.fev[i]
            t2=g.time()
            g.block.promote(self.cdst,dst,self.basis)
            t3=g.time()
            if verbose:
                g.message("Coarse-grid deflated in %g s (project %g s, coarse deflate %g s, promote %g s)" % 
                          (t3-t0,t1-t0,t2-t1,t3-t2))
            return self.inverter(matrix)(dst,src)

        return g.matrix_operator(mat = inv, inv_mat = matrix, otype = otype)
