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

    def __init__(self, mat, adj_mat = None, inv_mat = None, adj_inv_mat = None):
        self.mat = mat
        self.adj_mat = adj_mat
        self.inv_mat = inv_mat
        self.adj_inv_mat = adj_inv_mat

    def inv(self):
        return matrix_operator(self.inv_mat,self.adj_inv_mat,self.mat,self.adj_mat)

    def adj(self):
        return matrix_operator(self.adj_mat,self.mat,self.adj_inv_mat,self.inv_mat)

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
            dst=gpt.lattice(src)
        else:
            dst=first
            src=second
        self.mat(dst,src)
        return dst
