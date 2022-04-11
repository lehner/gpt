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
import gpt as g


class langevin_euler:
    @g.params_convention(epsilon=0.01)
    def __init__(self, rng, params):
        self.rng = rng
        self.eps = params["epsilon"]

    def __call__(self, fields, action):
        gr = action.gradient(fields, fields)
        for d, f in zip(gr, fields):
            f @= g.group.compose(
                -d * self.eps + self.rng.normal_element(g.lattice(d)) * (self.eps * 2.0) ** 0.5,
                f,
            )


# Phys. Rev. D 32, 2736 (1985); Phys. Rev. Lett. 55, 1854 (1985).
class langevin_bf:
    @g.params_convention(epsilon=0.01)
    def __init__(self, rng, params):
        self.rng = rng
        self.eps = params["epsilon"]

    def __call__(self, fields, action):
        gr = action.gradient(fields, fields)
        CA = gr[0].otype.CA
        sqrteps_eta = [
            g(self.rng.normal_element(g.lattice(d)) * (self.eps * 2.0) ** 0.5) for d in gr
        ]
        fields_tilde = g.copy(fields)
        for d, f, n in zip(gr, fields_tilde, sqrteps_eta):
            f @= g.group.compose(
                -d * self.eps - n,
                f,
            )
        gr_tilde = action.gradient(fields_tilde, fields_tilde)
        for d_tilde, d, f_tilde, f, n in zip(gr_tilde, gr, fields_tilde, fields, sqrteps_eta):
            f @= g.group.compose(
                -(d + d_tilde) * (self.eps * 0.5 * (1.0 + CA * self.eps / 6.0)) - n,
                f,
            )
