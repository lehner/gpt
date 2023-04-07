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
import gpt.qcd.fermion.domain

from gpt.qcd.fermion.register import register
from gpt.qcd.fermion.operator import (
    gauge_independent_g5_hermitian,
    differentiable_fine_operator,
    fine_operator,
    coarse_operator,
)
from gpt.qcd.fermion.boundary_conditions import *

# fine fermion operators
from gpt.qcd.fermion.zmobius import zmobius
from gpt.qcd.fermion.mobius import mobius
from gpt.qcd.fermion.wilson import wilson_clover, wilson_twisted_mass


# coarse-grid operator
import copy


@gpt.params_convention(make_hermitian=False, level=None)
def coarse_fermion(A, params):
    params = copy.deepcopy(params)  # save current parameters
    params["nbasis"] = A[0].otype.v_n1[0]
    return coarse_operator("coarse", A, params, otype=A[0].otype.vector_type)


# abbreviations / short-cuts
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
