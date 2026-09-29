#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-26  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.core.group import local_diffeomorphism, differentiable_functional


class local_stout(local_diffeomorphism):
    # checkerboarded stout smearing of the links in one direction, expressed
    # as a directional_parallel_transport of the plaquettes through the link
    # (AD path); one transport is built per grid and gauge group on first use
    @params_convention(dimension=None, checkerboard=None, rho=None)
    def __init__(self, params):
        self.params = params
        self.cache = {}

    def transport(self, fields):
        nd = fields[0].grid.nd
        U = fields[0:nd]
        key = (U[0].grid, U[0].otype.__name__)
        if key not in self.cache:
            mu = self.params["dimension"]
            rho = self.params["rho"]
            even, odd = g.even_odd_projectors(U[0].grid)
            P1 = even if self.params["checkerboard"] is g.even else odd
            description = [
                (rho, g.path().f(nu).f(mu).b(nu).b(mu)) for nu in range(nd) if nu != mu
            ] + [(rho, g.path().b(nu).f(mu).f(nu).b(mu)) for nu in range(nd) if nu != mu]
            self.cache[key] = g.qcd.gauge.smear.directional_parallel_transport(
                U, description, mu, g(even + odd), P1
            )
        return self.cache[key], U

    def __call__(self, fields):
        t, U = self.transport(fields)
        return t(U)

    def inv(self, fields, max_iter=100):
        t, U = self.transport(fields)
        return t.inv(U, max_iter=max_iter)

    def jacobian(self, fields, fields_prime, src):
        t, U = self.transport(fields)
        return t.jacobian(U, fields_prime[0 : len(U)], src)

    def log_det_jacobian(self, fields):
        t, U = self.transport(fields)
        return g(g.component.real(t.log_det_jacobian_field(U)))

    def action_log_det_jacobian(self):
        return local_stout_action_log_det_jacobian(self)


class local_stout_action_log_det_jacobian(differentiable_functional):
    def __init__(self, stout):
        self.stout = stout

    def __call__(self, fields):
        t, U = self.stout.transport(fields)
        return -t.log_det_jacobian(U).real

    def gradient(self, fields, dfields):
        t, U = self.stout.transport(fields)
        return t.action_log_det_jacobian_gradient(U, dfields)
