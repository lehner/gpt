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
#      P(U) = e^{ Re Tr Staple U } dU
#
#    using the metropolis algorithm.  This interface is not yet
#    stable.
import gpt as g


class metropolis:
    @g.params_convention(step_size=0.5, project_method="defect")
    def __init__(self, rng, params):
        self.rng = rng
        self.params = params

    def __call__(self, link, staple, mask):
        verbose = g.default.is_verbose(
            "metropolis"
        )  # need verbosity categories [ performance, progress ]
        project_method = self.params["project_method"]
        step_size = self.params["step_size"]

        number_accept = 0
        possible_accept = 0

        t = g.timer("metropolis")

        t("action")
        action = g.component.real(g.eval(-g.trace(link * g.adj(staple)) * mask))

        t("lattice")
        V = g.lattice(link)
        V_eye = g.identity(link)

        t("random")
        self.rng.element(V, scale=step_size, normal=True)

        t("update")
        V = g.where(mask, V, V_eye)

        link_prime = g.eval(V * link)
        action_prime = g.component.real(
            g.eval(-g.trace(link_prime * g.adj(staple)) * mask)
        )

        dp = g.component.exp(g.eval(action - action_prime))

        rn = g.lattice(dp)

        t("random")
        self.rng.uniform_real(rn)

        t("random")
        accept = dp > rn
        accept *= mask

        number_accept += g.norm2(accept)
        possible_accept += g.norm2(mask)

        link @= g.where(accept, link_prime, link)

        t()

        g.project(link, project_method)

        # g.message(t)
        if verbose:
            g.message(f"Metropolis acceptance rate: {number_accept / possible_accept}")
