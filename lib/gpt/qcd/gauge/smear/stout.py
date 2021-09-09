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
import numpy as np
import gpt as g
from gpt.params import params_convention


@params_convention(rho=None)
def stout_general(U, params):
    nd = len(U)
    C = g.qcd.gauge.staple_sum(U, params)
    U_prime = []
    for mu in range(nd):
        U_mu_prime = g(
            g.matrix.exp(
                g.qcd.gauge.project.traceless_anti_hermitian(C[mu] * g.adj(U[mu]))
            )
            * U[mu]
        )
        U_prime.append(U_mu_prime)
    return U_prime


@params_convention(rho=0.1, orthogonal_dimension=None)
def stout(U, params):
    nd = len(U)
    rho = params["rho"]
    orthogonal_dimension = params["orthogonal_dimension"]
    if orthogonal_dimension is None:
        orthogonal_dimension = -1
    rho_matrix = np.array(
        [
            [
                0.0
                if (
                    mu == orthogonal_dimension or nu == orthogonal_dimension or mu == nu
                )
                else rho
                for nu in range(nd)
            ]
            for mu in range(nd)
        ],
        dtype=np.float64,
    )
    return stout_general(U, rho=rho_matrix)
