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
#    Generate U with
#
#      P(U) = e^{ Re Tr adj(staple) U } dU
#
#    based on 1985 Kennedy and Pendleton paper (PLB 156 p393-399)
import gpt as g
import numpy as np
import sys


class su2_heat_bath:
    @g.params_convention(project_method="defect", niter=20)
    def __init__(self, rng, params):
        self.rng = rng
        self.params = params

    def __call__(self, link, staple, mask):
        verbose = g.default.is_verbose(
            "su2_heat_bath"
        )  # need verbosity categories [ performance, progress ]
        project_method = self.params["project_method"]

        # params
        niter = self.params["niter"]

        # temporaries
        grid = link.grid
        u2 = g.lattice(grid, g.ot_matrix_su_n_fundamental_group(2))
        u2_eye = g.identity(u2)
        one = g.identity(g.complex(grid))
        zero = g.complex(grid)
        zero[:] = 0
        eps = g.complex(grid)
        eps[:] = grid.precision.eps * 10.0
        xr = [g.complex(grid) for i in range(4)]
        a = [g.complex(grid) for i in range(4)]
        two_pi = g.complex(grid)
        two_pi[:] = 2.0 * np.pi
        accepted = g.complex(grid)
        d = g.complex(grid)
        V_eye = g.identity(link)

        # pauli
        pauli1, pauli2, pauli3 = tuple([g.lattice(u2) for i in range(3)])
        ident = g.identity(u2)
        pauli1[:] = 1j * np.array([[0, 1], [1, 0]], dtype=grid.precision.complex_dtype)
        pauli2[:] = 1j * np.array(
            [[0, 1j], [-1j, 0]], dtype=grid.precision.complex_dtype
        )
        pauli3[:] = 1j * np.array([[1, 0], [0, -1]], dtype=grid.precision.complex_dtype)

        # counter
        num_sites = round(g.norm2(g.where(mask, one, zero)))

        # shortcuts
        inv = g.component.pow(-1.0)

        # go through subgroups
        for subgroup in link.otype.su2_subgroups():

            V = g.eval(link * g.adj(staple))

            # extract u2 subgroup following Kennedy/Pendleton
            link.otype.block_extract(u2, V, subgroup)
            u2 @= u2 - g.adj(u2) + g.identity(u2) * g.trace(g.adj(u2))
            udet = g.matrix.det(u2)
            adet = g.component.abs(udet)
            nzmask = adet > eps
            u2 @= g.where(nzmask, u2, u2_eye)
            udet = g.where(nzmask, udet, one)
            xi = g.eval(0.5 * g.component.sqrt(udet))
            u2 @= 0.5 * u2 * inv(xi)

            # make sure that su2 subgroup projection worked
            assert g.group.defect(u2) < u2.grid.precision.eps * 10.0

            xi @= 2.0 * xi
            alpha = g.component.real(xi)

            # main loop
            it = 0
            num_accepted = 0
            accepted[:] = 0
            d[:] = 0
            while (num_accepted < num_sites) and (it < niter):
                self.rng.uniform_real(xr, min=0.0, max=1.0)

                xr[1] @= -g.component.log(xr[1]) * inv(alpha)
                xr[2] @= -g.component.log(xr[2]) * inv(alpha)
                xr[3] @= g.component.cos(g.eval(xr[3] * two_pi))
                xr[3] @= xr[3] * xr[3]

                xrsq = g.eval(xr[2] + xr[1] * xr[3])

                d = g.where(accepted, d, xrsq)

                thresh = g.eval(one - d * 0.5)
                xrsq @= xr[0] * xr[0]

                newly_accepted = g.where(xrsq < thresh, one, zero)
                accepted = g.where(
                    mask, g.where(newly_accepted, newly_accepted, accepted), zero
                )

                num_accepted = round(g.norm2(g.where(accepted, one, zero)))

                it += 1

            if verbose:
                g.message(f"SU(2)-heatbath update needed {it} / {niter} iterations")

            # update link
            a[0] @= g.where(mask, one - d, zero)

            a123mag = g.component.sqrt(g.component.abs(one - a[0] * a[0]))

            phi, cos_theta = g.complex(grid), g.complex(grid)
            self.rng.uniform_real([phi, cos_theta])
            phi @= phi * two_pi
            cos_theta @= (cos_theta * 2.0) - one
            sin_theta = g.component.sqrt(g.component.abs(one - cos_theta * cos_theta))

            a[1] @= a123mag * sin_theta * g.component.cos(phi)
            a[2] @= a123mag * sin_theta * g.component.sin(phi)
            a[3] @= a123mag * cos_theta

            ua = g.eval(a[0] * ident + a[1] * pauli1 + a[2] * pauli2 + a[3] * pauli3)

            b = g.where(mask, g.adj(u2) * ua, ident)
            link.otype.block_insert(V, b, subgroup)

            link @= g.where(accepted, V * link, link)

            # check
            check = g.where(accepted, ua * g.adj(ua) - ident, 0.0 * ident)
            delta = (g.norm2(check) / g.norm2(ident)) ** 0.5
            assert delta < grid.precision.eps * 10.0

            check = g.where(accepted, b * g.adj(b) - ident, 0.0 * ident)
            delta = (g.norm2(check) / g.norm2(ident)) ** 0.5
            assert delta < grid.precision.eps * 10.0

            check = g.where(accepted, V * g.adj(V) - V_eye, 0.0 * V_eye)
            delta = (g.norm2(check) / g.norm2(V_eye)) ** 0.5
            assert delta < grid.precision.eps * 10.0

        # project
        g.project(link, project_method)
