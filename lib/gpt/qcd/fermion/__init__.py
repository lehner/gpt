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
import gpt.qcd.fermion.coarse

from gpt.qcd.fermion.register import register
from gpt.qcd.fermion.operator import fine_operator, coarse_operator
from gpt.qcd.fermion.boundary_conditions import *

import copy

###
# instantiate fermion operators


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
    return fine_operator(
        "wilson_clover", U, params, otype=gpt.ot_vector_spin_color(4, 3)
    )


@gpt.params_convention(mass=None, cp=None, zeta=None, boundary_phases=None)
def rhq_columbia(U, params):
    return wilson_clover(
        U,
        mass=params["mass"],
        csw_r=params["cp"],
        csw_t=params["cp"],
        cF=1.0,
        xi_0=1.0,
        nu=params["zeta"],
        isAnisotropic=True,
        boundary_phases=params["boundary_phases"],
    )


@gpt.params_convention(
    omega=None, mass=None, b=None, c=None, M5=None, boundary_phases=None
)
def zmobius(U, params):
    params = copy.deepcopy(params)  # save current parameters
    params["Ls"] = len(params["omega"])
    return fine_operator("zmobius", U, params, otype=gpt.ot_vector_spin_color(4, 3))


@gpt.params_convention(
    mass=None, b=None, c=None, M5=None, boundary_phases=None, Ls=None
)
def mobius(U, params):
    params = copy.deepcopy(params)  # save current parameters
    return fine_operator("mobius", U, params, otype=gpt.ot_vector_spin_color(4, 3))


@gpt.params_convention(make_hermitian=False, level=None)
def coarse_fermion(A, params):
    params = copy.deepcopy(params)  # save current parameters
    params["nbasis"] = A[0].otype.v_n1[0]
    return coarse_operator("coarse", A, params, otype=A[0].otype.vector_type)
