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

    def diagonal_jacobian_gradient(self, fields, fields_prime, left, right):
        # Compute \partial_rho [ left . (\partial f_mu / \partial U_mu) . right ]
        # for each field rho, i.e. the gradient of the mu->mu Jacobian block
        # bilinearly contracted with the fixed test function `left` (output)
        # and direction `right` (input).
        #
        # Build left . (\partial f_mu/\partial U_mu) . right as a 1-deep functional
        # over (fields, left, right) and take its 1-deep gradient.  The mu->mu
        # block is obtained with the "apply Jacobian to a direction" reverse pass
        # on a 2-deep graph (a 1-deep node seed), which yields a 1-deep
        # 1st-derivative graph; differentiating that 1-deep graph is a true 1st
        # derivative, so the result is correctly differentiable.
        #
        # (The earlier 2-deep implementation did the 2nd reverse directly on the
        # "apply Jacobian to a direction" graph, whose U-dependence is not the
        # correctly trivialized one, giving a wrong derivative.)
        #
        # The nested reverse applies the infinitesimal_to_cartesian symmetrization
        # 0.5*(x + adj(x)) at the group leaf, which halves the derivative of the
        # 1st-derivative graph; the factor of 2 below compensates for it.
        rad = g.ad.reverse

        mu = self.mu
        N = len(fields)
        assert len(fields) == N

        _U = [rad.node(g.copy(u)) for u in fields]
        _left = rad.node(g.copy(left), with_gradient=False)
        _right = rad.node(g.copy(right), with_gradient=False)

        # 1-deep transform graph (for the seed) and 2-deep graph (for the reverse)
        _Up = self.ft(_U)
        aU = [rad.node(_U[i]) for i in range(N)]
        aUft = self.ft(aU)

        seed = g.cartesian_to_infinitesimal(_Up[mu], _right)
        aUft[mu](initial_gradient=seed)          # 1st reverse: 2-deep -> 1-deep graph
        J_right_mu = aU[mu].gradient             # 1-deep graph = (\partial f_mu/\partial U_mu) . right

        act = g.inner_product(_left, rad.node(self.P1, with_gradient=False) * J_right_mu)
        func = act.functional(*(_U + [_left, _right]))
        grads = func.gradient(fields + [left, right], fields)   # 1-deep gradient w.r.t. _U

        # resolve to plain lattices so the (expensive) nested node graphs are
        # released before the caller accumulates over the generators.
        # non-contributing fields (links not in the path) have a None gradient;
        # replace those with zero.
        from gpt.ad.reverse.util import is_node, value_of
        def _res(x):
            if x is None:
                return None
            while is_node(x):
                x = value_of(x)
            if isinstance(x, g.expr):
                x = g(x)
            return x
        zero = g(0 * left)
        out = []
        for x in grads:
            r = _res(x)
            out.append(g(zero) if r is None else g(2.0 * r))
        return out

    def action_log_det_jacobian_gradient(self, fields, dfields):
        # The mu->mu block M (see jacobian_matrix) satisfies M[a,b] = (d f_mu/d U_mu)[b,a],
        # i.e. M is the transpose of the Jacobian block J in the (output, input) basis.
        # The action is -log det(J) = -log det(M), so
        #   \partial_rho (-log det M) = -tr(M^{-1} dM/drho)
        #                              = -sum_{a,b} M^{-1}[a,b] (dM/drho)[a,b].
        # Each term is \partial_rho <left, M.right> with left = P1*gen[a] and
        # right = -sum_b M^{-1}[a,b] gen[b] (the P1-masked site-dependent vector),
        # evaluated by diagonal_jacobian_gradient.

        J = self.jacobian_matrix(fields)
        Jinv = g.matrix.inv(J)

        fields_prime = self(fields)

        grid = fields[0].grid
        dt = grid.precision.complex_dtype
        otype = fields[0].otype
        otype_cartesian = otype.cartesian()
        generators = otype_cartesian.generators(dt)
        right = g.group.cartesian(fields[0])
        left = g.group.cartesian(fields[0])

        Jinv = g.separate_color(Jinv)

        for a in range(len(generators)):
            left @= self.P1 * generators[a]
            right @= sum(-Jinv[a, b] * generators[b] for b in range(len(generators)))
            right = g.where(self.P1, right, g(0*left))
            gr = self.diagonal_jacobian_gradient(fields, fields_prime, left, right)
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
