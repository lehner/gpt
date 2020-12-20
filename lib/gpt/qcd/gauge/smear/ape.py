#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Lorenzo Barca    (lorenzo1.barca@ur.de)
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
from gpt.core.suN_utils import project_onto_suN

@params_convention(alpha=2.5, orthogonal_dimension=3, Blk_Max=None, Blk_Accuracy=None)
def ape(U, params):
    nd = len(U)
    alpha = params["alpha"]
    orthogonal_dimension = params["orthogonal_dimension"]
    blk_max = params["blk_max"]
    blk_accuracy = params["blk_accuracy"]
    rho_matrix = np.array(
        [
            [
                0.0
                if (
                    mu == orthogonal_dimension or nu == orthogonal_dimension or mu == nu
                 )
                else 1.0
                for nu in range(nd)
            ]
            for mu in range(nd)
        ],
        dtype=np.float64,
    )
    return ape_general(U, rho=rho_matrix, blk_max=blk_max, blk_accuracy=blk_accuracy, orthogonal_dimension=orthogonal_dimension)

@params_convention(alpha=2.5)
def ape_general(U, params):
#   Following (27) of https://arxiv.org/pdf/hep-lat/0505012.pdf
#   here the smearing factor alpha is multiplying the starting links
    nd = len(U)
    alpha = params["alpha"]
    orthogonal_dimension = params["orthogonal_dimension"]
    sum_staples = g.qcd.gauge.staples(U, params)
    U_smear = []
    for mu in range(nd):
        U_mu_smear = U[mu]
        if mu!= orthogonal_dimension:
             U_mu_smear =  g(U[mu] * alpha) + g(sum_staples[mu])
             U_unproj = g.eval(g.adj(U_mu_smear))
             # project onto suN
             U_mu_smear = project_onto_suN(U_unproj, U[mu], params)
             # Reunitarize
             g.qcd.reunitize(U_mu_smear)
             assert U_mu_smear.otype.is_element(U_mu_smear)
        U_smear.append(U_mu_smear)
    return U_smear
