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
from gpt.qcd.gauge.smear.differentiable import dft_diffeomorphism
from .differentiable import assert_compatible
from gpt.core.group import differentiable_functional


class directional_parallel_transport(dft_diffeomorphism):
    def __init__(self, U, description_mu, mu, P0=None, P1=None, parameters=[]):
        self.description_mu = description_mu
        self.mu = mu
        self.P0 = P0
        self.P1 = P1

        parameter_indices = [
            parameters.index(weight) if weight in parameters else None
            for weight, path in description_mu
        ]
        nd = len(U)
        np = len(parameters)
        ntot = nd + np
        fields = U + parameters

        cache = {}

        def ft(xU):
            assert len(xU) == ntot

            cache_key = f"{type(xU[0])}_depth{g.ad.reverse.util.value_depth(xU[0])}"
            if cache_key not in cache:
                paths = [y[1] for y in description_mu]
                cache[cache_key] = g.parallel_transport(xU[0:nd], paths)

            pt = cache[cache_key]
            if P0 is not None:
                xU_P0 = [g(xU[i] * P0) if i == mu else xU[i] for i in range(nd)]
            else:
                xU_P0 = xU[0:nd]

            xparams = xU[nd:]
            assert len(xparams) == np

            sU = list(pt(xU_P0))
            sm = None
            idx = 0
            for weight, path in description_mu:
                if weight in parameters:
                    assert not isinstance(weight, g.ad.reverse.node_base)
                    weight = xparams[parameter_indices[idx]]
                xp = g(weight * sU[idx])
                if sm is None:
                    sm = xp
                else:
                    sm += xp
                idx += 1

            assert sm is not None

            if P1 is not None:
                if isinstance(sm, g.ad.reverse.node_base):
                    # a matrix x scalar node-graph breaks the trace contraction
                    # downstream; mask with a where to keep the matrix otype
                    sm = g.where(P1, sm, g.ad.reverse.node(sm._container.zero(), with_gradient=False))
                else:
                    sm *= P1

            sm = g(g.matrix.exp(g.qcd.gauge.project.traceless_anti_hermitian(sm)) * xU[mu])
            return [sm if i == mu else xU[i] for i in range(nd)] + xparams

        super().__init__(fields, ft)

    def diagonal_jacobian(self, fields, fields_prime, dfields_mu):
        mu = self.mu
        N = len(fields_prime)
        assert len(fields) == N
        aU_prime_mu = g.cartesian_to_infinitesimal(fields_prime[mu], dfields_mu)
        for nu in range(len(fields)):
            assert_compatible(self.aU[nu].value, fields[nu])
            self.aU[nu].value = fields[nu]
        self.aUft[mu](initial_gradient=aU_prime_mu)
        self.aU[mu].gradient.otype = dfields_mu.otype
        return g(self.aU[mu].gradient * self.P1)

    def jacobian_matrix(self, fields):
        fields_prime = self(fields)
        grid = fields[0].grid
        dt = grid.precision.complex_dtype
        otype = fields[0].otype
        otype_cartesian = otype.cartesian()
        generators = otype_cartesian.generators(dt)
        src = g.group.cartesian(fields[0])
        M = g.lattice(grid, g.ot_matrix_su_n_adjoint_algebra(otype.Nc))

        for a in range(len(generators)):
            src @= self.P1 * generators[a]
            dst = self.diagonal_jacobian(fields, fields_prime, src)
            coor = otype_cartesian.coordinates(dst)
            for b in range(len(generators)):
                M[:, :, :, :, a, b] = coor[b][:]
        return M

    def log_det_jacobian(self, fields):
        M = self.jacobian_matrix(fields)
        M_det = g.matrix.det(M)
        M_log_det = g.component.log(M_det)
        zero = g.lattice(M_log_det)
        zero[:] = 0
        M_log_det = g.where(self.P1, M_log_det, zero)
        return g.sum(M_log_det)

    def action_log_det_jacobian(self):
        return dpt_action_log_det_jacobian(self)

    def diagonal_jacobian_gradient(self, fields, fields_prime, a, right):
        # Compute gen_a . (d_U J_x) . right  on-site (P1-masked), where
        # J_x = d_U f[mu] | on-site is the site-local Jacobian block.
        # Option (a) recipe (2-deep + 1 contraction, the validated HVP shape):
        #   pass 1: scalar S = sum(P1 * Tr[f[mu] * gen_a])  (default seed;
        #            gen_a is a constant gpt_object generator, so the 1st
        #            derivative is the a-row of the on-site Jacobian block)
        #   pass 2: group.inner_product(nRight, aaU[mu].gradient)()  (HVP in
        #            the pointwise `right` direction)
        #   -> aaU[mu].value.gradient = gen_a . (d_U J_x) . right
        rad = g.ad.reverse

        mu = self.mu
        N = len(fields_prime)
        assert len(fields) == N

        otype = fields[0].otype
        otype_cart = otype.cartesian()
        generators = otype_cart.generators(fields[0].grid.precision.complex_dtype)
        gen_a = generators[a]

        aaU = [rad.node(u) for u in self.aU]

        for nu in range(len(aaU)):
            aaU[nu].value.value = fields[nu]

        aaUft = self.ft(aaU)

        # pass 1: scalar functional (default seed), gen_a baked in as a constant
        t = g.trace(aaUft[mu] * gen_a)
        S = g.sum(t * self.P1)
        S()

        # pass 2: HVP contraction in the right direction
        nRight = rad.node(right, with_gradient=False)
        ip = g.group.inner_product(nRight, aaU[mu].gradient)
        ip()

        # resolve to plain lattices so the (expensive) nested node graphs are
        # released before the caller accumulates over the generators
        from gpt.ad.reverse.util import is_node, value_of
        def _res(x):
            while is_node(x):
                x = value_of(x)
            if isinstance(x, g.expr):
                x = g(x)
            return x
        return [_res(aaU[nu].value.gradient) for nu in range(len(aaU))]

    def action_log_det_jacobian_gradient(self, fields, dfields):
        # det(J_{ab} + drho_c \partial_{rho_c} J_{ab}) = det(J) (1 + J^-1_{ba} drho_c \partial_{rho_c} J_{ab})
        # -> \partial_{rho_c} det(J) = det(J) J^-1_{ba} \partial_{rho_c} J_{ab}
        # \partial -\log \det(J) = -1/det(J) \partial det(J)
        # Compute tr[\partial_rho (\partial_U f) M]

        J = self.jacobian_matrix(fields)
        Jinv = g.matrix.inv(J)

        fields_prime = self(fields)

        grid = fields[0].grid
        dt = grid.precision.complex_dtype
        otype = fields[0].otype
        otype_cartesian = otype.cartesian()
        generators = otype_cartesian.generators(dt)
        right = g.group.cartesian(fields[0])

        Jinv = g.separate_color(Jinv)

        for a in range(len(generators)):
            right @= sum(-Jinv[b, a] * generators[b] for b in range(len(generators)))
            gr = self.diagonal_jacobian_gradient(fields, fields_prime, a, right)
            if a == 0:
                gr_sum = gr
            else:
                for nu in range(len(gr)):
                    gr_sum[nu] += gr[nu]

        return [gr_sum[fields.index(d)] for d in dfields]


class dpt_action_log_det_jacobian(differentiable_functional):
    def __init__(self, parent):
        self.parent = parent

    def __call__(self, fields):
        return -self.parent.log_det_jacobian(fields).real

    def gradient(self, fields, dfields):
        return self.parent.action_log_det_jacobian_gradient(fields, dfields)
