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
import numpy as np
from gpt.qcd.gauge.smear.differentiable import dft_diffeomorphism
from .differentiable import assert_compatible
from gpt.core.group import differentiable_functional


def _staple_description(description_mu, mu, nd):
    # If every path is a closed loop at x whose ONLY traversal of the updated
    # link (x, mu) is its final b(mu) step, the transported loop factorizes as
    # C(x) U_mu(x)^dag with a staple C that does not depend on U_mu(x).  The
    # mu->mu Jacobian block is then a site-local function of (U_mu(x), C(x)),
    # which is what the local log-det-Jacobian code below exploits.  Returns
    # the staple description (paths with the final b(mu) removed), or None if
    # the factorization does not apply.
    staple = []
    for weight, p in description_mu:
        steps = [(nu, 1 if d > 0 else -1) for nu, d in p.path for _ in range(abs(d))]
        pos = [0] * nd
        touched = []
        for i, (nu, s) in enumerate(steps):
            if nu == mu and ((s == 1 and all(x == 0 for x in pos)) or (
                s == -1 and all(pos[k] == (1 if k == mu else 0) for k in range(nd))
            )):
                touched.append(i)
            pos[nu] += s
        if any(x != 0 for x in pos) or touched != [len(steps) - 1] or steps[-1] != (mu, -1):
            return None
        head = list(p.path[:-1])
        nu, d = p.path[-1]
        if d != -1:
            head.append((nu, d + 1))
        staple.append((weight, g.path(head)))
    return staple


def _adjoint_matrix(grid, Nc, coor):
    # M[a, b] = coor[a, b] as one adjoint-algebra matrix field; a single
    # (cached) copy plan on the fields' memory instead of per-component
    # host-side slice assignments
    M = g.lattice(grid, g.ot_matrix_su_n_adjoint_algebra(Nc))
    return g.merge_color(M, coor)


class _generator_kernels:
    # site-local tensor kernels between algebra fields (N x N) and
    # adjoint-algebra matrices (ng x ng), with the generators T_b as constant
    # fields (read as kernel temporaries, as in local_stout):
    #   rows(M, l)    M[a, b] = tr(l_a T_b) / tr(T_b T_b)   (coordinates of l_a)
    #   combine(r, K) r_a     = -sum_b K[a, b] T_b
    # one kernel each instead of ng^2 (coordinates) or ng^2 (products) kernels
    def __init__(self, grid, otype_cartesian):
        N = otype_cartesian.shape[0]
        generators = otype_cartesian.generators(grid.precision.complex_dtype)
        ng = len(generators)
        norm = [complex(np.trace(t.array @ t.array)) for t in generators]
        # (the scale is applied once per element, so all norms must agree)
        assert all(abs(n - norm[0]) < 10 * grid.precision.eps for n in norm)
        ti = g.stencil.tensor_instructions
        self.N, self.ng = N, ng
        self.fgenerators = [g.lattice(grid, otype_cartesian) for _ in range(ng)]
        for f, t in zip(self.fgenerators, generators):
            f[:] = t
        nonzero = [
            [(i, j) for i in range(N) for j in range(N) if abs(t.array[i, j]) != 0.0]
            for t in generators
        ]

        # fields: M, l_0..l_{ng-1}, T_0..T_{ng-1}
        code = []
        for a in range(ng):
            for b in range(ng):
                for k, (j, i) in enumerate(nonzero[b]):
                    code.append(
                        (0, a * ng + b, ti.mov if k == 0 else ti.inc, 1.0,
                         [(1 + a, 0, i * N + j), (-(1 + ng + b), 0, j * N + i)])
                    )
                code.append((0, a * ng + b, ti.mul, 1.0 / norm[0], [(0, 0, a * ng + b)]))
        M = g.lattice(grid, g.ot_matrix_su_n_adjoint_algebra(N))
        self._rows = g.stencil.tensor(M, [(0,) * grid.nd], code, [(len(code), 1)])

        # fields: r_0..r_{ng-1}, K, T_0..T_{ng-1}
        code = []
        for a in range(ng):
            first = {}
            for b in range(ng):
                for i, j in nonzero[b]:
                    code.append(
                        (a, i * N + j, ti.inc if (i, j) in first else ti.mov, 1.0,
                         [(ng, 0, a * ng + b), (-(ng + 1 + b), 0, i * N + j)])
                    )
                    first[i, j] = True
            # (every element is covered by some generator)
            assert len(first) == N * N
            for e in range(N * N):
                code.append((a, e, ti.mul, -1.0, [(a, 0, e)]))
        self._combine = g.stencil.tensor(self.fgenerators[0], [(0,) * grid.nd], code, [(len(code), 1)])

    def rows(self, M, l):
        self._rows(M, *l, *self.fgenerators)
        return M

    def combine(self, r, K):
        self._combine(*r, K, *self.fgenerators)
        return r


def _get_generator_kernels(grid, otype_cartesian):
    # one set per (grid, algebra type), kept on the grid
    cache = grid.__dict__.setdefault("_dpt_generator_kernels", {})
    key = otype_cartesian.__name__
    if key not in cache:
        cache[key] = _generator_kernels(grid, otype_cartesian)
    return cache[key]


class directional_parallel_transport(dft_diffeomorphism):
    def __init__(
        self, U, description_mu, mu, P0=None, P1=None, parameters=[], loop_function=None
    ):
        # loop_function: an optional site-local map f(sm, xparams) of the
        # weighted loop sum, U_mu' = exp(TA(P1 f(sm, xparams))) U_mu (default:
        # the identity), where xparams are the parameters (plain fields or
        # nodes, in the order of `parameters`).  It must work on plain fields
        # and on nodes of any depth (node-first products: sm * p, not p * sm)
        # and be gauge covariant (products of sm and adj(sm), traces).
        self.description_mu = description_mu
        self.mu = mu
        self.P0 = P0
        self.P1 = P1
        self.parameters = parameters
        self.loop_function = loop_function

        nd = len(U)
        np = len(parameters)
        ntot = nd + np
        self.nd = nd
        fields = U + parameters

        cache = {}

        def ft(xU):
            assert len(xU) == ntot
            sm = self._weighted_transport(cache, description_mu, xU)
            sm = self._update(sm, xU[mu], xU[nd:])
            return [sm if i == mu else xU[i] for i in range(nd)] + xU[nd:]

        self.description_staple = _staple_description(description_mu, mu, nd)
        self._staple_cache = {}
        self._vjp = None

        super().__init__(fields, ft)

    def _weighted_transport(self, cache, description, xU):
        # sum_k weight_k * transport_k(U), with P0 applied to U_mu; works for
        # plain fields and for nodes of any depth
        nd, mu, P0, parameters = self.nd, self.mu, self.P0, self.parameters

        # static: may be handed computed nodes, and resolving their depth
        # here would cache a value that node.forward then reuses instead of
        # rebuilding it from the updated leaves
        cache_key = f"{type(xU[0])}_depth{g.ad.reverse.util.value_depth_static(xU[0])}"
        if cache_key not in cache:
            # paths sharing a weight are summed inside the stencil, so the
            # staple sum below costs one multiply per distinct weight
            # instead of one multiply and one add per path
            cache[cache_key] = g.parallel_transport_weighted(xU[0:nd], description)

        pt, group_info = cache[cache_key]
        if P0 is not None:
            xU_P0 = [g(xU[i] * P0) if i == mu else xU[i] for i in range(nd)]
        else:
            xU_P0 = xU[0:nd]

        xparams = xU[nd:]
        assert len(xparams) == len(parameters)

        sU = list(pt(xU_P0))
        assert len(sU) == len(group_info)
        terms = []
        for k, (_key, weight) in enumerate(group_info):
            if weight is None:
                # numeric weight: already folded into the kernel
                # coefficient, so sU[k] IS the accumulated staple sum
                xp = sU[k]
            else:
                if any(weight is p for p in parameters):
                    assert not isinstance(weight, g.ad.reverse.node_base)
                    weight = xparams[g.util.index_by_identity(parameters, weight)]
                xp = g(weight * sU[k])
            terms.append(xp)

        assert terms
        # never accumulate in place: a folded-weight term IS the stencil's
        # own target field, and mutating it would corrupt the stencil output
        sm = terms[0]
        for t in terms[1:]:
            sm = sm + t
        return sm

    def _update(self, sm, xU_mu, xparams):
        # U_mu' = exp(TA(P1 f(sm))) U_mu
        return g(g.matrix.exp(self._project(sm, xparams)) * xU_mu)

    def _staple(self, xfields):
        # the weighted staple C with transported loop = C U_mu^dag (P0 applied)
        return g(self._weighted_transport(self._staple_cache, self.description_staple, xfields))

    def _local_ft(self, xU_mu, xC, xparams):
        # the site-local map U_mu' = F(U_mu, C, params) for a fixed staple C
        xU_P0 = g(xU_mu * self.P0) if self.P0 is not None else xU_mu
        return self._update(g(xC * g.adj(xU_P0)), xU_mu, xparams)

    def jacobian(self, fields, fields_prime, dfields):
        # only output mu is transformed, every other output is the identity
        # map, whose Jacobian passes its direction through: one reverse pass
        # (for output mu) instead of one per output
        mu = self.mu
        N = len(fields)
        assert len(fields_prime) == N and len(dfields) == N
        seed = g.cartesian_to_infinitesimal(fields_prime[mu], dfields[mu])
        if self.description_staple is not None:
            grads = self._local_vjp(fields, seed)
        else:
            for nu in range(N):
                assert_compatible(self.aU[nu].value, fields[nu])
                self.aU[nu].value = fields[nu]
                self.aU[nu].zero_gradient()
            self.aUft[mu](initial_gradient=seed)
            grads = [self.aU[nu].gradient for nu in range(N)]
        gradient = []
        for nu in range(N):
            gr = grads[nu]
            if nu != mu:
                gr = g.copy(dfields[nu]) if gr is None else g(gr + dfields[nu])
            elif gr is None:
                gr = g(0 * dfields[nu])
            if isinstance(gr, g.lattice):
                gr.otype = dfields[nu].otype
            gradient.append(gr)
        return gradient

    def _local_vjp(self, fields, seed):
        # the VJP of output mu through the site-local map f(U_mu, C) and the
        # staple C (see _staple_description): the staple stencil and its
        # adjoint carry one link fewer per path than the transported loop
        rad = g.ad.reverse
        mu = self.mu
        # the graphs are built once and re-used with new leaf values: node
        # graphs are reference cycles, so graphs built per call would only be
        # released by the cyclic garbage collector, which does not see the
        # size of the fields they hold
        plain = not any(isinstance(x, rad.node_base) for x in fields)
        nd = self.nd
        if plain and self._vjp is not None:
            nodes, aC, _U, _C, _P, aF = self._vjp
            for n, x in zip(nodes + _P, fields + fields[nd:]):
                assert_compatible(n.value, x)
                n.value = x
        else:
            nodes = [rad.node(x) for x in fields]
            aC = self._staple(nodes)
            _U = rad.node(fields[mu])
            # a plain matrix (not a group element): no conversion to the algebra
            _C = rad.node(g.lattice(fields[mu]), infinitesimal_to_cartesian=False)
            # separate parameter leaves for the local map (the staple pass
            # below would reset gradients deposited in shared leaves)
            _P = [rad.node(x) for x in fields[nd:]]
            aF = self._local_ft(_U, _C, _P)
            if plain:
                self._vjp = (nodes, aC, _U, _C, _P, aF)
        # the roots keep their values after a reverse pass
        aC.value = None
        aF.value = None
        # one staple forward, shared by the local pass and the staple's reverse
        _C.value = aC(with_gradients=False, retain_values=True)
        _U.value = fields[mu]
        aF(initial_gradient=seed)
        grad_U, grad_C = _U.gradient, _C.gradient
        grad_P = [n.gradient for n in _P]
        aC(initial_gradient=grad_C)
        grads = [n.gradient for n in nodes]
        for i, gr in [(mu, grad_U)] + [(nd + i, gr) for i, gr in enumerate(grad_P)]:
            if gr is not None:
                grads[i] = gr if grads[i] is None else g(grads[i] + gr)
        # release the fields held by the graph until the next call
        aC.value = None
        aF.value = None
        _C.value = None
        for n in nodes + [_U, _C] + _P:
            n.gradient = None
            n._borrowed.clear()
        return grads

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
        if self.description_staple is not None:
            return self._local_jacobian_matrix(fields)[0]
        return self._jacobian_matrix_generic(fields)

    def _local_jacobian_matrix(self, fields):
        # the mu->mu block from VJPs through the stencil-free local map
        # f(U_mu, C) at fixed staple C (see _staple_description); returns the
        # block and the staple
        rad = g.ad.reverse
        mu, P1 = self.mu, self.P1
        C = self._staple(fields)
        U_mu = fields[mu]

        grid = U_mu.grid
        otype = U_mu.otype
        otype_cartesian = otype.cartesian()
        generators = otype_cartesian.generators(grid.precision.complex_dtype)
        # one forward shared by the 8 reverse passes
        aU = rad.node(U_mu)
        xparams = [rad.node(x, with_gradient=False) for x in fields[self.nd :]]
        aUft = self._local_ft(aU, rad.node(C, with_gradient=False), xparams)
        U_prime_mu = aUft(with_gradients=False, retain_values=True)
        src = g.group.cartesian(U_mu)
        rows = []
        for a in range(len(generators)):
            src @= P1 * generators[a]
            aUft(initial_gradient=g.cartesian_to_infinitesimal(U_prime_mu, src), retain_values=True)
            aU.gradient.otype = src.otype
            rows.append(g(aU.gradient * P1))
        # M[a, b] = coordinate b of row a, all in one kernel
        M = g.lattice(grid, g.ot_matrix_su_n_adjoint_algebra(otype.Nc))
        return _get_generator_kernels(grid, otype_cartesian).rows(M, rows), C

    def _jacobian_matrix_generic(self, fields):
        fields_prime = self(fields)
        grid = fields[0].grid
        dt = grid.precision.complex_dtype
        otype = fields[0].otype
        otype_cartesian = otype.cartesian()
        generators = otype_cartesian.generators(dt)
        src = g.group.cartesian(fields[0])
        coor = {}
        for a in range(len(generators)):
            src @= self.P1 * generators[a]
            dst = self.diagonal_jacobian(fields, fields_prime, src)
            for b, c in enumerate(otype_cartesian.coordinates(dst)):
                coor[a, b] = c
        return _adjoint_matrix(grid, otype.Nc, coor)

    def inv(self, fields, max_iter=100):
        # with nodes: the preimage as a node (its backward from the Jacobian
        # of this transport, see g.ad.reverse.preimage)
        if any(isinstance(x, g.ad.reverse.node_base) for x in fields):
            (u,) = g.ad.reverse.preimage(self, fields, [self.mu], lambda v: self.inv(v, max_iter))
            return [u if i == self.mu else fields[i] for i in range(len(fields))]
        # invert U_mu' = exp(TA(P1 f(C U_mu^dag))) U_mu by the fixed-point
        # iteration U_mu <- exp(-TA(P1 f(C U_mu^dag))) U_mu'.  The staple C is
        # evaluated once on the smeared fields: this requires that it does
        # not depend on the updated links (e.g. a checkerboard P1 with
        # plaquette staples), which is verified at the end.
        assert self.description_staple is not None
        mu, nd = self.mu, self.nd
        C = self._staple(fields)
        xparams = fields[nd:]
        U_prime_mu = fields[mu]
        U_mu = g.copy(U_prime_mu)
        eps = U_mu.grid.precision.eps
        for it in range(max_iter):
            U_mu_last = g.copy(U_mu)
            xU_P0 = g(U_mu * self.P0) if self.P0 is not None else U_mu
            U_mu @= g.matrix.exp(-self._project(g(C * g.adj(xU_P0)), xparams)) * U_prime_mu
            eps2 = g.norm2(U_mu_last - U_mu) / U_mu.grid.gsites
            if eps2 < eps**2:
                break
        if it == max_iter - 1:
            g.message(
                f"Warning: directional_parallel_transport could not be inverted; last eps^2 = {eps2} after {max_iter} iterations"
            )
            return None
        U = [U_mu if i == mu else fields[i] for i in range(len(fields))]
        # the staple of the result must be the one used above (where it
        # enters the update)
        C_check = self._staple(U)
        eps2 = g.norm2(self.P1 * (C_check - C)) / max(g.norm2(self.P1 * C), 1e-300)
        assert eps2 < 1e4 * eps**2, f"staple depends on the updated links ({eps2})"
        return U

    def _project(self, sm, xparams):
        # TA(P1 f(sm, params)), the generator of the update; every path (the
        # graph of ft, the site-local map and inv) builds its update here
        if self.loop_function is not None:
            sm = g(self.loop_function(sm, xparams))
        if self.P1 is not None:
            sm = g(sm * self.P1)
        return g.qcd.gauge.project.traceless_anti_hermitian(sm)

    def log_det_jacobian_field(self, fields):
        # the site-local log det of the mu->mu block (zero outside P1)
        M = self.jacobian_matrix(fields)
        M_det = g.matrix.det(M)
        M_log_det = g.component.log(M_det)
        zero = g.lattice(M_log_det)
        zero[:] = 0
        return g.where(self.P1, M_log_det, zero)

    def log_det_jacobian(self, fields):
        return g.sum(self.log_det_jacobian_field(fields))

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
        # (a field without gradient gets a zero of its own type: links in the
        # algebra, parameters as they are)
        out = []
        for i, x in enumerate(grads):
            r = _res(x)
            if r is None:
                r = g(0 * (left if i < self.nd else fields[i]))
            out.append(g(2.0 * r))
        return out

    def action_log_det_jacobian_gradient(self, fields, dfields):
        if self.description_staple is not None:
            return self._local_action_log_det_jacobian_gradient(fields, dfields)
        return self._action_log_det_jacobian_gradient_generic(fields, dfields)

    def _local_action_log_det_jacobian_gradient(self, fields, dfields):
        # Same contraction as the generic version below, but the Jacobian
        # block is a site-local function of (U_mu, C): the 8 second-order
        # passes run over the stencil-free local map f(U_mu, C) and yield
        # gradients w.r.t. U_mu and C.  The C gradient is linear, so it is
        # summed over the generators first and pushed through the staple
        # stencil by ONE first-order reverse pass.
        rad = g.ad.reverse
        from gpt.ad.reverse.util import is_node, value_of

        def _res(x):
            while is_node(x):
                x = value_of(x)
            return g(x)

        mu, P1 = self.mu, self.P1
        M, C = self._local_jacobian_matrix(fields)

        U_mu = fields[mu]
        otype_cartesian = U_mu.otype.cartesian()
        generators = otype_cartesian.generators(U_mu.grid.precision.complex_dtype)
        ng = len(generators)
        # right_a = -P1 sum_b Jinv[a, b] T_b for all a in one kernel; M
        # vanishes outside P1, so its inverse is masked with a where (a mask
        # multiplication would turn the non-finite entries there into nan)
        # (the ng x ng matrices are 7x a color matrix for SU(3) and are not
        # needed in the passes below: release each as soon as possible)
        Jinv = g.matrix.inv(M)
        del M
        zero = g.lattice(Jinv)
        zero[:] = 0
        Jinv = g.where(P1, Jinv, zero)
        del zero
        right = [g.lattice(U_mu.grid, otype_cartesian) for _ in range(ng)]
        _get_generator_kernels(U_mu.grid, otype_cartesian).combine(right, Jinv)
        del Jinv
        left = g.group.cartesian(U_mu)
        P1_node = rad.node(P1, with_gradient=False)

        # 1-deep leaves (never modified, no copies needed); the staple leaf is
        # a plain matrix (not a group element), so its gradient is not
        # converted to the algebra
        _U = rad.node(U_mu)
        _C = rad.node(C, infinitesimal_to_cartesian=False)
        _P = [rad.node(x) for x in fields[self.nd :]]

        # 2-deep "apply Jacobian block to right" (see diagonal_jacobian_gradient);
        # the forward does not depend on the generator, so it runs once and
        # its (1-deep) values are shared by all passes.  The seed is built
        # from the forward's own (retained) 1-deep value, so the derivative
        # graph and the seed share its nodes.
        aU = rad.node(_U)
        aUft = self._local_ft(
            aU, rad.node(_C, with_gradient=False), [rad.node(x, with_gradient=False) for x in _P]
        )
        _Up = aUft(with_gradients=False, retain_values=True)

        grad_U = None
        grad_C = None
        grad_P = [None] * len(_P)
        for a in range(ng):
            left @= P1 * generators[a]
            _left = rad.node(left, with_gradient=False)
            _right = rad.node(right[a], with_gradient=False)

            # both reverse passes retain the shared forward values; the
            # per-generator part of the graph is released with act
            aUft(
                initial_gradient=g.cartesian_to_infinitesimal(_Up, _right),
                retain_values=True,
            )
            act = g.inner_product(_left, P1_node * aU.gradient)
            act(retain_values=True)
            gU, gC = _res(_U.gradient), _res(_C.gradient)
            grad_U = gU if grad_U is None else g(grad_U + gU)
            grad_C = gC if grad_C is None else g(grad_C + gC)
            for i, n in enumerate(_P):
                if n.gradient is not None:
                    gP = _res(n.gradient)
                    grad_P[i] = gP if grad_P[i] is None else g(grad_P[i] + gP)
                    del gP

            del act, _left, _right, gU, gC
            right[a] = None

        del aU, aUft, _Up, _U, _C, _P
        del left, right, C, P1_node

        # chain rule through the staple; the factor 2 is the one explained in
        # diagonal_jacobian_gradient
        nodes = [rad.node(x) for x in fields]
        seed = g(2.0 * grad_C)
        del grad_C
        self._staple(nodes)(initial_gradient=seed)
        del seed
        out = []
        for i, n in enumerate(nodes):
            r = g(0 * fields[i]) if n.gradient is None else g(n.gradient)
            if i == mu:
                r = g(r + 2.0 * grad_U)
            elif i >= self.nd and grad_P[i - self.nd] is not None:
                r = g(r + 2.0 * grad_P[i - self.nd])
            out.append(r)
        return [out[g.util.index_by_identity(fields, d)] for d in dfields]

    def _action_log_det_jacobian_gradient_generic(self, fields, dfields):
        # The mu->mu block M (see jacobian_matrix) satisfies M[a,b] = (d f_mu/d U_mu)[b,a],
        # i.e. M is the transpose of the Jacobian block J in the (output, input) basis.
        # The action is -log det(J) = -log det(M), so
        #   \partial_rho (-log det M) = -tr(M^{-1} dM/drho)
        #                              = -sum_{a,b} M^{-1}[a,b] (dM/drho)[a,b].
        # Each term is \partial_rho <left, M.right> with left = P1*gen[a] and
        # right = -sum_b M^{-1}[a,b] gen[b] (the P1-masked site-dependent vector),
        # evaluated by diagonal_jacobian_gradient.

        J = self._jacobian_matrix_generic(fields)
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

        return [gr_sum[g.util.index_by_identity(fields, d)] for d in dfields]

class dpt_action_log_det_jacobian(differentiable_functional):
    def __init__(self, parent):
        self.parent = parent

    def __call__(self, fields):
        return -self.parent.log_det_jacobian(fields).real

    def gradient(self, fields, dfields):
        return self.parent.action_log_det_jacobian_gradient(fields, dfields)
