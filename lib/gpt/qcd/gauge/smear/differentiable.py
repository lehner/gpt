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
from gpt.ad import reverse as rad


def assert_compatible(a, b, tag=""):
    if type(a) is not type(b):
        raise Exception(f"Incompatible types: {type(a)} and {type(b)}{tag}")
    if isinstance(a, rad.node_base):
        assert_compatible(a.value, b.value, " root " + str(type(a)))


class dft_diffeomorphism(diffeomorphism):
    def __init__(self, U, ft):
        self.ft = ft
        self.aU = [rad.node(u) for u in U]
        self.aUft = ft(self.aU)

    def __call__(self, fields):
        # ft needs to be callable with a node or a lattice
        res = self.ft(fields)
        return [g(x) for x in res]

    def jacobian(self, fields, fields_prime, dfields):
        N = len(fields_prime)
        assert len(fields) == N
        assert len(dfields) == N
        aU_prime = [g(2j * dfields[mu] * fields_prime[mu]) for mu in range(N)]
        for mu in range(N):
            assert_compatible(self.aU[mu].value, fields[mu])
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

    def adjoint_jacobian(self, fields, dfields):
        N = len(fields)
        assert len(dfields) == N

        fad = g.ad.forward
        eps = fad.infinitesimal("eps")

        On = fad.landau(eps**2)
        aU = [fad.series(fields[mu], On) for mu in range(N)]
        for mu in range(N):
            aU[mu][eps] = g(1j * dfields[mu] * fields[mu])

        aU_prime = self.ft(aU)

        gradient = [g(aU_prime[mu][eps] * g.adj(aU_prime[mu][1]) / 1j) for mu in range(N)]

        return gradient


class dft_action_log_det_jacobian(differentiable_functional):
    def __init__(self, U, ft, dfm, dfm_node, inverter_force, inverter_action):
        self.dfm = dfm
        self.dfm_node = dfm_node
        self.inverter_force = inverter_force
        self.inverter_action = inverter_action
        self.N = len(U)
        mom = [g.group.cartesian(u) for u in U]

        _U = [rad.node(g.copy(u)) for u in U]
        _left = [rad.node(g.copy(u), with_gradient=False) for u in mom]
        _right = [rad.node(g.copy(u), with_gradient=False) for u in mom]
        _Up = dfm_node(_U)
        J_right = dfm_node.jacobian(_U, _Up, _right)

        act = None
        for mu in range(self.N):
            if mu == 0:
                act = g.inner_product(_left[mu], J_right[mu])
            else:
                act = g(act + g.inner_product(_left[mu], J_right[mu]))

        self.left_J_right = act.functional(*(_U + _left + _right))

    def mat_J(self, U, U_prime):
        def _mat(dst_5d, src_5d):
            src = g.separate(src_5d, dimension=0)
            dst = self.dfm.jacobian(U, U_prime, src)
            dst_5d @= g.merge(dst, dimension=0)

        return _mat

    def mat_Jdag(self, U):
        def _mat(dst_5d, src_5d):
            src = g.separate(src_5d, dimension=0)
            dst = self.dfm.adjoint_jacobian(U, src)
            dst_5d @= g.merge(dst, dimension=0)

        return _mat

    def __call__(self, fields):
        U = fields[0 : self.N]
        mom = fields[self.N :]
        mom_xd = g.merge(mom, dimension=0)
        U_prime = [g(x) for x in self.dfm.ft(U)]
        mom_prime_xd = self.inverter_action(self.mat_J(U, U_prime))(mom_xd)
        mom_prime2_xd = self.inverter_action(self.mat_Jdag(U))(mom_prime_xd)
        return g.inner_product(mom_xd, mom_prime2_xd).real

    def gradient(self, fields, dfields):
        U = fields[0 : self.N]
        mom = fields[self.N :]

        # all eigenvalues of J need to be positive because for trivial FT they are all 1 and we
        # stay invertible so none of them can have crossed a zero

        # M = J J^dag
        # ->  S = pi^dag J^{-1}^dag J^{-1} pi
        # -> dS = -pi^dag J^{-1}^dag dJ^dag J^{-1}^dag J^{-1} pi
        #         -pi^dag J^{-1}^dag J^{-1} dJ J^{-1} pi
        #       = -pi^dag J^{-1}^dag J^{-1} dJ J^{-1} pi + C.C.  ->  need two inverse of J
        assert dfields == U

        U_prime = [g(x) for x in self.dfm.ft(U)]

        mom_xd = g.merge(mom, dimension=0)
        mom_prime_xd = self.inverter_force(self.mat_J(U, U_prime))(mom_xd)
        mom_prime2_xd = self.inverter_force(self.mat_Jdag(U))(mom_prime_xd)

        mom_prime2 = g.separate(mom_prime2_xd, dimension=0)
        mom_prime = g.separate(mom_prime_xd, dimension=0)

        gradients = self.left_J_right.gradient(U + mom_prime2 + mom_prime, U)
        return [g(-x - g.adj(x)) for x in gradients]

    def draw(self, fields, rng):
        U = fields[0 : self.N]
        mom = fields[self.N :]
        assert len(mom) == self.N
        assert len(U) == self.N

        rng.normal_element(mom, scale=1.0)
        U_prime = [g(x) for x in self.dfm.ft(U)]

        mom_prime = self.dfm.jacobian(U, U_prime, mom)

        act = 0.0
        for mu in range(self.N):
            act += g.norm2(mom[mu])
            mom[mu] @= mom_prime[mu]

        return act


class differentiable_field_transformation:
    def __init__(self, U, ft, inverter_force, inverter_action, optimizer):
        self.ft = ft
        self.U = U
        self.dfm = dft_diffeomorphism(self.U, self.ft)
        self.dfm_node = dft_diffeomorphism([rad.node(u) for u in self.U], self.ft)
        self.inverter_force = inverter_force
        self.inverter_action = inverter_action
        self.optimizer = optimizer

    def diffeomorphism(self):
        return self.dfm

    def inverse(self, Uft):
        aU = [rad.node(g.copy(u)) for u in Uft]
        aUft_target = [rad.node(u, with_gradient=False) for u in Uft]
        aUft = self.ft(aU)
        fnc = sum([g.norm2(aUft_target[mu] - aUft[mu]) for mu in range(len(Uft))]).functional(*aU)
        U = g.copy(Uft)
        self.optimizer(fnc)(U, U)
        return U

    def action_log_det_jacobian(self):
        return dft_action_log_det_jacobian(
            self.U, self.ft, self.dfm, self.dfm_node, self.inverter_force, self.inverter_action
        )
