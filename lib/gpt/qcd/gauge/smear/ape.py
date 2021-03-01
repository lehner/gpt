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
    t_total = - gpt.time()

    t_setstaple, t_addstaple, t_project, t_reunitize, t_su3check = 0.0, 0.0, 0.0, 0.0, 0.0
    nd = len(u)

    # create mask for staples
    params["rho"] = np.ones((nd, nd), dtype=np.float64)
    for mu in range(nd):
        for nu in range(nd):
            if mu == params["orthogonal_dimension"] or nu == params["orthogonal_dimension"] or mu == nu:
                params["rho"][mu, nu] = 0.0

    # create staples
    t_setstaple -= gpt.time()
    staplesum = gpt.qcd.gauge.staple_sum(u, params)
    t_setstaple += gpt.time()

    u_apesmeared = []
    for mu in range(nd):
        gpt.message(f"Starting direction {mu}...")
        # start with original link
        u_tmp = gpt.copy(u[mu])
        if mu != params["orthogonal_dimension"]:
            # get the unprojected, i.e., u + staples
            t_addstaple -= gpt.time()
            u_unprojected = gpt.eval(gpt.adj(params["alpha"] * u_tmp + staplesum[mu]))
            t_addstaple += gpt.time()

            # take the original gauge field as a starting point for the projections
            t_project -= gpt.time()
            project_onto_suN(u_tmp, u_unprojected, params)
            t_project += gpt.time()

            # reunitarize
            t_reunitize -= gpt.time()
            gpt.qcd.reunitize(u_tmp)
            t_reunitize += gpt.time()

            # check if result is su3
            t_su3check -= gpt.time()
            u_tmp.otype.is_element(u_tmp)
            t_su3check += gpt.time()

        u_apesmeared.append(u_tmp)
        gpt.message(f"Direction {mu} done.")

    t_total += gpt.time()

    if gpt.default.is_verbose("ape"):
        t_profiled = t_setstaple + t_addstaple + t_project + t_reunitize + t_su3check
        t_unprofiled = t_total - t_profiled

        gpt.message("ape: total", t_total, "s")
        gpt.message("ape: t_setstaple", t_setstaple, "s", round(100 * t_setstaple / t_total, 1), "%")
        gpt.message("ape: t_addstaple", t_addstaple, "s", round(100 * t_addstaple / t_total, 1), "%")
        gpt.message("ape: t_project", t_project, "s", round(100 * t_project / t_total, 1), "%")
        gpt.message("ape: t_reunitize", t_reunitize, "s", round(100 * t_reunitize / t_total, 1), "%")
        gpt.message("ape: t_su3check", t_su3check, "s", round(100 * t_su3check / t_total, 1), "%")
        gpt.message("ape: unprofiled", t_unprofiled, "s", round(100 * t_unprofiled / t_total, 1), "%")

    return u_apesmeared
