#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-24  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.core.group import diffeomorphism, differentiable_functional


class dft_diffeomorphism(diffeomorphism):
    def __init__(self, U, ft):
        rad = g.ad.reverse
        self.ft = ft
        self.aU = [rad.node(u.new()) for u in U]
        self.aUft = ft(self.aU)

    def __call__(self, fields):
        res = self.ft(fields)
        return [g(x) for x in res]

    def jacobian(self, fields, fields_prime, dfields):
        N = len(fields_prime)
        assert len(fields) == N
        assert len(fields) == N
        assert len(fields) == N
        aU_prime = [g(2j * dfields[mu] * fields_prime[mu]) for mu in range(N)]
        for mu in range(N):
            self.aU[mu].value = fields[mu]
        gradient = [None] * N
        for mu in range(N):
            self.aUft[mu](initial_gradient=aU_prime[mu])
            for nu in range(N):
                if gradient[nu] is None:
                    gradient[nu] = self.aU[nu].gradient
                else:
                    gradient[nu] = g(gradient[nu] + self.aU[nu].gradient)

        return gradient


class dft_action_log_det_jacobian(differentiable_functional):
    def __init__(self, U, ft, dfm, inverter):
        self.dfm = dfm
        self.inverter = inverter
        self.N = len(U)
        mom = [g.group.cartesian(u) for u in U]
        rad = g.ad.reverse

        _U = [rad.node(g.copy(u)) for u in U]
        _mom = [rad.node(g.copy(u)) for u in mom]
        _Up = dfm(_U)
        momp = dfm.jacobian(_U, _Up, _mom)

        act = None
        for mu in range(self.N):
            if mu == 0:
                act = g.norm2(momp[mu])
            else:
                act = g(act + g.norm2(momp[mu]))

        self.action = act.functional(*(_U + _mom))

    def __call__(self, fields):
        return self.action(fields)

    def gradient(self, fields, dfields):
        return self.action.gradient(fields, dfields)

    def draw(self, fields, rng):
        U = fields[0 : self.N]
        mom = fields[self.N :]
        assert len(mom) == self.N
        assert len(U) == self.N

        rng.normal_element(mom, scale=1.0)

        U_prime = self.dfm(U)

        def _mat(dst_5d, src_5d):
            src = g.separate(src_5d, dimension=0)
            dst = self.dfm.jacobian(U, U_prime, src)
            dst_5d @= g.merge(dst, dimension=0)

        mom_xd = g.merge(mom, dimension=0)

        mom_prime_xd = self.inverter(_mat)(mom_xd)
        mom_prime = g.separate(mom_prime_xd, dimension=0)

        act = 0.0
        for mu in range(self.N):
            act += g.norm2(mom[mu])
            mom[mu] @= mom_prime[mu]

        return act


class differentiable_field_transformation:
    def __init__(self, U, ft, inverter, optimizer):
        self.ft = ft
        self.U = U
        self.dfm = dft_diffeomorphism(self.U, self.ft)
        self.inverter = inverter
        self.optimizer = optimizer

    def diffeomorphism(self):
        return self.dfm

    def inverse(self, Uft):
        rad = g.ad.reverse
        aU = [rad.node(g.copy(u)) for u in Uft]
        aUft_target = [rad.node(u, with_gradient=False) for u in Uft]
        aUft = self.ft(aU)
        fnc = sum([g.norm2(aUft_target[mu] - aUft[mu]) for mu in range(len(Uft))]).functional(*aU)
        U = g.copy(Uft)
        self.optimizer(fnc)(U, U)
        return U

    def action_log_det_jacobian(self):
        return dft_action_log_det_jacobian(self.U, self.ft, self.dfm, self.inverter)
