#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2021  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


class projected_matrix_operator:
    def __init__(self, mat, adj_mat, grid, otype, parity):
        self.mat = mat
        self.adj_mat = adj_mat
        self.grid = grid
        self.otype = otype
        self.parity = parity

    def adj(self):
        return projected_matrix_operator(
            self.adj_mat,
            self.mat,
            tuple(reversed(self.grid)),
            tuple(reversed(self.otype)),
            self.parity,
        )

    def __call__(self, left, right):
        return self.mat(g(left), g(right))
