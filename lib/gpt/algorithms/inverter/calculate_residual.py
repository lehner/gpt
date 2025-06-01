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


class calculate_residual:
    def __init__(self, tag=None, threshold_residual=None):
        self.tag = "" if tag is None else f"{tag}: "
        self.threshold_residual = threshold_residual

    def __call__(self, mat):
        def inv(dst, src):
            for i in range(len(dst)):
                eps = g.norm2(mat * dst[i] - src[i]) ** 0.5
                nrm = g.norm2(src[i]) ** 0.5
                if nrm != 0.0:
                    g.message(
                        f"{self.tag}| mat * dst[{i}] - src[{i}] | / | src | = {eps / nrm}, | src[{i}] | = {nrm}"
                    )
                    if self.threshold_residual is not None:
                        assert eps / nrm <= self.threshold_residual
                else:
                    g.message(
                        f"{self.tag}| mat * dst[{i}] - src[{i}] | = {eps}, | src[{i}] | = {nrm}"
                    )
                    if self.threshold_residual is not None:
                        assert eps <= self.threshold_residual

        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            vector_space=vector_space,
            accept_guess=(True, False),
            accept_list=True,
        )
