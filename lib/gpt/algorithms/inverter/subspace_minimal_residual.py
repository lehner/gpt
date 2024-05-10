#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  Implementing: https://arxiv.org/pdf/hep-lat/9509012.pdf
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
from gpt.algorithms import base


class subspace_minimal_residual(base):
    def __init__(self, solution_space):
        super().__init__()
        self.solution_space = solution_space

    def __call__(self, mat):
        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space
        else:
            mat = g.matrix_operator(mat=mat)

        @self.timed_function
        def inv(psi, src, t):
            if len(self.solution_space) == 0:
                return

            t("orthonormalize")
            v = g.orthonormalize(g.copy(self.solution_space))

            # Idea is to minimize
            #
            #  res = | M a_i v_i - src |^2
            #      = v_i^dag a_i^dag M^dag M a_j v_j + src^dag src - src^dag M a_i v_i - v_i^dag a_i^dag M^dag src
            #
            # by selecting an optimal a_i, i.e., to compute
            #
            #  d res/d a_i^dag = v_i^dag M^dag M a_j v_j - v_i^dag M^dag src = 0
            #
            # Therefore
            #
            #  G_ij a_j = b_i
            #
            # with b_i = v_i^dag M^dag src,  G_ij = v_i^dag M^dag M v_j
            #

            t("mat v")
            mat_v = [mat(x) for x in v]

            t("projected source")
            b = g.inner_product(mat_v, src)[:, 0]

            t("projected matrix")
            G_ij = np.matrix([g.inner_product(mat_v, mat_v[j])[:, 0] for j in range(len(v))]).T

            t("solve")
            a = np.linalg.solve(G_ij, b)

            t("linear combination")
            g.linear_combination(psi, v, a)

            eps2 = g.norm2(mat(psi) - src) / g.norm2(src)
            self.log(
                f"minimal residual with {len(v)}-dimensional solution space has eps^2 = {eps2}"
            )

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            vector_space=vector_space,
            accept_guess=(True, False),
        )
