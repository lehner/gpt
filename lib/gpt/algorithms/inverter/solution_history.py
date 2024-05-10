#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.algorithms import base


class solution_history(base):
    def __init__(self, solution_space, inverter, N):
        super().__init__()
        self.inverter = inverter
        self.N = N
        self.solution_space = solution_space

    def __call__(self, mat):
        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space

        inv_mat = self.inverter(mat)

        @self.timed_function
        def inv(psi, src, t):
            t("inverter")
            inv_mat(psi, src)

            t("update")
            space = self.solution_space
            if len(space) == self.N:
                space.pop()
            space.insert(0, psi)

            self.log(f"solution space now has {len(self.solution_space)} / {self.N} elements")

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            vector_space=vector_space,
            accept_guess=(True, False),
        )
