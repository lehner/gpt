#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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


class wilson_class_operator(
    differentiable_fine_operator, gauge_independent_g5_hermitian
):
    def __init__(self, name, U, params, otype=None):
        differentiable_fine_operator.__init__(self, name, U, params, otype)

        def _G5(dst, src):
            dst @= gpt.gamma[5] * src

        gauge_independent_g5_hermitian.__init__(
            self, gpt.matrix_operator(_G5, grid=self.F_grid, otype=otype)
        )


@gpt.params_convention(
    kappa=None,
    mass=None,
    cF=1,
    use_legacy=False,
    boundary_phases=None,
    isAnisotropic=None,
    csw_r=None,
    csw_t=None,
    nu=None,
    xi_0=None,
)
def wilson_clover(U, params):
    params = copy.deepcopy(params)  # save current parameters
    if params["kappa"] is not None:
        assert params["mass"] is None
        params["mass"] = 1.0 / params["kappa"] / 2.0 - 4.0
        del params["kappa"]
    if params["use_legacy"]:
        assert params["boundary_phases"][-1] != 0.0  # only new op supports open bc
    if params["boundary_phases"][-1] != 0.0:
        assert params["cF"] == 1.0  # forbid usage of cF without open bc
    if params["csw_r"] == 0.0 and params["csw_t"] == 0.0:
        # for now Grid does not have MooeeDeriv for clover term
        operator_class = wilson_class_operator
    else:
        operator_class = fine_operator
    return operator_class(
        "wilson_clover", U, params, otype=gpt.ot_vector_spin_color(4, 3)
    )
