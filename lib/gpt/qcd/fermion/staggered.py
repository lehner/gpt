#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt, copy
from gpt.qcd.fermion.operator import (
    gauge_independent_g5_hermitian,
    differentiable_fine_operator,
    fine_operator,
)


class staggered_operator(differentiable_fine_operator, gauge_independent_g5_hermitian):
    def __init__(self, name, U, params, otype=None, daggered=False):
        differentiable_fine_operator.__init__(self, name, U, params, otype, daggered)

        def _G5(dst, src):
            dst @= gpt.gamma[5] * src

        gauge_independent_g5_hermitian.__init__(
            self, gpt.matrix_operator(_G5, vector_space=self.vector_space)
        )


@gpt.params_convention(mass=None, c1=None, c2=None, u0=None)
def staggered(U, params):
    params = copy.deepcopy(params)  # save current parameters
    return staggered_operator("staggered", U, params, otype=gpt.ot_vector_color(3))
