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
import gpt
from gpt.params import params_convention
from gpt.core.suNutils import project_onto_suN

@params_convention(alpha=2.5, orthogonal_dimension=3, max_iterations=20, accuracy=1e-20)
def ape(u, params):
    nd = len(u)

    # create mask for staples
    params["rho"] = np.ones((nd, nd), dtype=np.float64)
    for mu in range(nd):
        for nu in range(nd):
            if mu == params["orthogonal_dimension"] or nu == params["orthogonal_dimension"] or mu == nu:
                params["rho"][mu, nu] = 0.0

    # create staples
    staplesum = gpt.qcd.gauge.smear.staple_sum(u, params)

    u_apesmeared = []
    for mu in range(nd):
        gpt.message(f"Starting direction {mu}...")
        # start with original link
        u_tmp = u[mu]
        if mu != params["orthogonal_dimension"]:
            # get the unprojected, i.e., u + staples
            u_unprojected = gpt.eval(gpt.adj(gpt(u_tmp * params["alpha"]) + gpt(staplesum[mu])))

            # take the original gauge field as a starting point for the projections
            project_onto_suN(u_tmp, u_unprojected, params)

            # reunitarize
            g.qcd.reunitize(U_mu_smear)
            g.qcd.gauge.assert_unitary(U_mu_smear)

        u_apesmeared.append(u_tmp)
        gpt.message(f"Direction {mu} done.")
    return u_apesmeared
