#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.params import params_convention
from gpt.core.group import local_diffeomorphism
from gpt.qcd.gauge.smear.directional_frozen_shift import directional_frozen_shift, zero_functional


class frozen_shift(local_diffeomorphism):
    # checkerboarded volume-preserving shift of the links in one direction by
    # a generator of the frozen links (see directional_frozen_shift, of which
    # this is a thin wrapper with the interface of local_stout); the fields
    # are the links followed by the generator's parameters; one transform is
    # built per grid and gauge group on first use
    @params_convention(dimension=None, checkerboard=None, generator=None, parameters=[])
    def __init__(self, params):
        self.params = params
        self.cache = {}

    def transform(self, fields):
        nd = fields[0].grid.nd
        n = nd + len(self.params["parameters"])
        U = fields[0:n]
        key = (U[0].grid, U[0].otype.__name__)
        if key not in self.cache:
            even, odd = g.even_odd_projectors(U[0].grid)
            P1 = even if self.params["checkerboard"] is g.even else odd
            self.cache[key] = directional_frozen_shift(
                U[0:nd],
                self.params["dimension"],
                P1,
                self.params["generator"],
                parameters=self.params["parameters"],
            )
        return self.cache[key], U

    def __call__(self, fields):
        t, U = self.transform(fields)
        return t(U)

    def inv(self, fields, max_iter=None, eps=None):
        t, U = self.transform(fields)
        return t.inv(U)

    def jacobian(self, fields, fields_prime, src, inputs=None):
        t, U = self.transform(fields)
        return t.jacobian(U, fields_prime[0 : len(U)], src, inputs=inputs)

    def log_det_jacobian(self, fields):
        t, U = self.transform(fields)
        return t.log_det_jacobian_field(U)

    def action_log_det_jacobian(self):
        return zero_functional()

    def check_frozen(self, fields, rng=None, eps=None):
        t, U = self.transform(fields)
        return t.check_frozen(U, rng, eps)
