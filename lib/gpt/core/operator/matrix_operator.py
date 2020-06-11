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
import gpt
from gpt.core.expr import factor

# A^dag (A^-1)^dag = (A^-1 A)^dag = 1^\dag = 1 ->
# (A^dag)^-1 = (A^-1)^dag
class matrix_operator(factor):

    def __init__(self, mat, adj_mat = None, inv_mat = None, adj_inv_mat = None,
                 otype = None, zero_lhs = False, grid_lhs = None):

        self.mat = mat
        self.adj_mat = adj_mat
        self.inv_mat = inv_mat
        self.adj_inv_mat = adj_inv_mat

        # matrices act on otype, this allows for automatic application of tensor versions
        # also should handle lists of lattices
        self.otype = otype

        # do we request the lhs of lhs = A rhs to be initialized to zero
        # if it is not given?
        self.zero_lhs = zero_lhs

        # does the lhs of lhs = A rhs live on a different grid?
        self.grid_lhs = grid_lhs

    def inv(self):
        return matrix_operator(self.inv_mat, self.adj_inv_mat, self.mat, self.adj_mat,
                               otype=self.otype, zero_lhs=self.zero_lhs, grid_lhs=self.grid_lhs)

    def adj(self):
        return matrix_operator(self.adj_mat, self.mat, self.adj_inv_mat, self.inv_mat,
                               otype=self.otype, zero_lhs=self.zero_lhs, grid_lhs=self.grid_lhs)

    def unary(self, u):
        if u == gpt.factor_unary.BIT_TRANS|gpt.factor_unary.BIT_CONJ:
            return self.adj()
        elif u == gpt.factor_unary.NONE:
            return self
        assert(0)

    def __call__(self, first, second = None):
        assert(not self.mat is None)

        if second is None:

            src=first
            if self.grid_lhs is None:
                dst=gpt.lattice(src)
            else:
                dst=gpt.lattice(self.grid_lhs, self.otype)

            if self.zero_lhs:
                dst[:]=0
        else:
            dst=first
            src=second

        if self.otype is None or self.otype.__name__ == first.otype.__name__:
            self.mat(dst,src)
        else:
            self.otype.distribute(self.mat, dst, src, zero_lhs = self.zero_lhs)

        return dst
