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
import gpt.qcd.fermion.reference
import gpt.qcd.fermion.preconditioner

from gpt.qcd.fermion.register import register
from gpt.qcd.fermion.operator import fine_operator, coarse_operator

import copy

###
# instantiate fermion operators


@gpt.params_convention()
def wilson_clover(U, params):
    params = copy.deepcopy(params)  # save current parameters
    if "kappa" in params:
        assert "mass" not in params
        params["mass"] = 1.0 / params["kappa"] / 2.0 - 4.0
        del params["kappa"]
    if "use_legacy" not in params:
        params["use_legacy"] = False  # default to new, faster implementation
    return fine_operator(
        "wilson_clover", U, params, otype=gpt.ot_vector_spin_color(4, 3)
    )


@gpt.params_convention()
def rhq_columbia(U, params):
    return wilson_clover(
        U,
        mass=params["mass"],
        csw_r=params["cp"],
        csw_t=params["cp"],
        xi_0=1.0,
        nu=params["zeta"],
        isAnisotropic=True,
        boundary_phases=params["boundary_phases"],
    )


@gpt.params_convention()
def zmobius(U, params):
    params = copy.deepcopy(params)  # save current parameters
    assert "Ls" not in params
    params["Ls"] = len(params["omega"])
    return fine_operator("zmobius", U, params, otype=gpt.ot_vector_spin_color(4, 3))


@gpt.params_convention()
def mobius(U, params):
    params = copy.deepcopy(params)  # save current parameters
    return fine_operator("mobius", U, params, otype=gpt.ot_vector_spin_color(4, 3))


@gpt.params_convention(make_hermitian=False)
def coarse(A, params):
    params = copy.deepcopy(params)  # save current parameters
    assert "nbasis" not in params
    params["nbasis"] = A[0].otype.v_n1[0]
    return coarse_operator("coarse", A, params, otype=A[0].otype.vector_type)
