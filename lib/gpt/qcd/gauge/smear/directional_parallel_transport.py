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
from gpt.ad.reverse.util import resolve


def _masked(P1, x, fill=0.0):
    # x on the sites of P1, fill elsewhere (a where: a mask multiplication
    # would turn non-finite entries outside P1 into nan)
    z = g.lattice(x)
    z[:] = fill
    return g.where(P1, x, z)


def _add(acc, x):
    # accumulate, starting from None
    return x if acc is None else g(acc + x)


def _with(fields, i, x):
    # fields with element i replaced by x
    return [x if j == i else f for j, f in enumerate(fields)]


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
        self.nd = nd
        fields = U + parameters

        cache = {}

        def ft(xU):
            assert len(xU) == nd + len(parameters)
            sm = self._weighted_transport(cache, description_mu, xU)
            return _with(xU, mu, self._update(sm, xU[mu], xU[nd:]))

        self.description_staple = g.staple_description(description_mu, mu, nd)
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

    def _local_generator(self, xU_mu, xC, xparams):
        # TA(P1 f(C U_mu^dag)) for a fixed staple C (P0 applied to U_mu)
        xU_P0 = g(xU_mu * self.P0) if self.P0 is not None else xU_mu
        return self._project(g(xC * g.adj(xU_P0)), xparams)

    def _local_ft(self, xU_mu, xC, xparams):
        # the site-local map U_mu' = F(U_mu, C, params) for a fixed staple C
        return g(g.matrix.exp(self._local_generator(xU_mu, xC, xparams)) * xU_mu)

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
            self._set_leaves(fields)
            for leaf in self.aU:
                leaf.zero_gradient()
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
        # staple C (see g.staple_description): the staple stencil and its
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
                grads[i] = _add(grads[i], gr)
        # release the fields held by the graph until the next call
        aC.value = None
        aF.value = None
        _C.value = None
        for n in nodes + [_U, _C] + _P:
            n.gradient = None
            n._borrowed.clear()
        return grads

    def jacobian_matrix(self, fields, staple=None):
        # the mu->mu block M[a, b] = coordinate b of the VJP of output mu with
        # the generator a (masked to P1); staple(fields): a staple other than
        # the transport's own (local transports only), e.g. prescribed sample
        # staples
        if self.description_staple is not None:
            return self._local_jacobian_matrix(fields, None if staple is None else g(staple(fields)))[0]
        assert staple is None
        return self._jacobian_matrix_generic(fields)[0]

    def _block(self, U_mu, U_prime_mu, vjp):
        # M from the VJPs (vjp(seed) -> gradient w.r.t. U_mu) of the generators
        P1 = self.P1
        otype_cartesian = U_mu.otype.cartesian()
        kernels = g.group.algebra_kernels(U_mu.grid, otype_cartesian)
        src = g.group.cartesian(U_mu)
        rows = []
        for T in kernels.fgenerators:
            src @= P1 * T
            gr = vjp(g.cartesian_to_infinitesimal(U_prime_mu, src))
            gr.otype = src.otype
            rows.append(g(gr * P1))
        M = g.lattice(U_mu.grid, g.ot_matrix_su_n_adjoint_algebra(U_mu.otype.Nc))
        return kernels.rows(M, rows)

    def _local_jacobian_matrix(self, fields, C=None):
        # the block from VJPs through the stencil-free local map f(U_mu, C)
        # at fixed staple C (default: the staple of fields, see
        # g.staple_description), one forward shared by the passes; returns the
        # block and the staple
        rad = g.ad.reverse
        if C is None:
            C = self._staple(fields)
        U_mu = fields[self.mu]
        aU = rad.node(U_mu)
        xparams = [rad.node(x, with_gradient=False) for x in fields[self.nd :]]
        aUft = self._local_ft(aU, rad.node(C, with_gradient=False), xparams)
        U_prime_mu = aUft(with_gradients=False, retain_values=True)

        def vjp(seed):
            aUft(initial_gradient=seed, retain_values=True)
            return aU.gradient

        return self._block(U_mu, U_prime_mu, vjp), C

    def _jacobian_matrix_generic(self, fields):
        # the block from VJPs through the full transport graph; returns the
        # block and the transported fields
        mu = self.mu
        fields_prime = self(fields)
        self._set_leaves(fields)

        def vjp(seed):
            self.aUft[mu](initial_gradient=seed)
            return self.aU[mu].gradient

        return self._block(fields[mu], fields_prime[mu], vjp), fields_prime

    def inv(self, fields, max_iter=100, eps=None, newton_rate=0.5):
        # with nodes: the preimage as a node (its backward from the Jacobian
        # of this transport, see g.ad.reverse.preimage)
        if any(isinstance(x, g.ad.reverse.node_base) for x in fields):
            (u,) = g.ad.reverse.preimage(
                self, fields, [self.mu], lambda v: self.inv(v, max_iter, eps, newton_rate)
            )
            return _with(fields, self.mu, u)
        # invert U_mu' = exp(TA(P1 f(C U_mu^dag))) U_mu by the fixed-point
        # iteration U_mu <- exp(-TA(P1 f(C U_mu^dag))) U_mu'.  The staple C is
        # evaluated once on the smeared fields: this requires that it does
        # not depend on the updated links (e.g. a checkerboard P1 with
        # plaquette staples), which is verified at the end.  If the iteration
        # contracts slowly (a rate above newton_rate, None: never), it
        # continues with site-local Newton steps (see _newton_step), which
        # also converge where the fixed-point iteration does not (a site map
        # that is a bijection but not a contraction).  Converged when the
        # change per site is below eps (default: ten times the precision,
        # above the rounding floor of the update); raises otherwise (see
        # g.algorithms.nonlinear.fixed_point, whose history and number of
        # Newton steps are kept as inverse_history and
        # inverse_newton_iterations).
        assert self.description_staple is not None
        mu, nd = self.mu, self.nd
        C = self._staple(fields)
        xparams = fields[nd:]
        U_prime_mu = fields[mu]
        precision = U_prime_mu.grid.precision.eps

        def step(x):
            x[0] @= g.matrix.exp(g(-1.0 * self._local_generator(x[0], C, xparams))) * U_prime_mu

        newton_step = self._newton(fields, U_prime_mu, C, xparams)
        fp = g.algorithms.nonlinear.fixed_point(
            eps=10 * precision if eps is None else eps,
            maxiter=max_iter,
            accelerate_rate=newton_rate,
        )
        name = "the inverse of directional_parallel_transport"
        try:
            (U_mu,) = fp([g.copy(U_prime_mu)], step, newton_step, name=name)
        finally:
            self.inverse_history = fp.history
            self.inverse_newton_iterations = fp.accelerated_iterations
        U = _with(fields, mu, U_mu)
        if self.inverse_newton_iterations > 0:
            # Newton finds a preimage, which is the preimage only if the site
            # map is a bijection; a solution on the other side of a fold has
            # det M <= 0 (the map of a bijection connected to the identity
            # has det M > 0 everywhere)
            det = _masked(self.P1, g.matrix.det(self._local_jacobian_matrix(U, C)[0]), 1.0)[:].real
            if det.min() <= 0:
                raise RuntimeError(
                    f"directional_parallel_transport could not be inverted: the Newton solution has "
                    f"det M <= 0 at {int((det <= 0).sum())} sites (the site map is not a bijection)"
                )
        # the staple of the result must be the one used above (where it
        # enters the update)
        C_check = self._staple(U)
        eps2 = g.norm2(self.P1 * (C_check - C)) / max(g.norm2(self.P1 * C), 1e-300)
        assert eps2 < 1e4 * precision**2, f"staple depends on the updated links ({eps2})"
        return U

    def _inverse_directions(self, M, U_mu, transpose=False, weight=None):
        # right_a = -sum_b K[a, b] T_b with K = M^-1 (transposed, times a
        # weight field) on P1 and zero elsewhere (M vanishes outside P1), for
        # all a in one kernel.  (The ng x ng matrices are 7x a color matrix
        # for SU(3): M is released by the caller as soon as possible.)
        K = g.matrix.inv(M)
        if transpose:
            K = g(g.transpose(K))
        K = _masked(self.P1, K)
        if weight is not None:
            K = g(weight * K)
        otype_cartesian = U_mu.otype.cartesian()
        kernels = g.group.algebra_kernels(U_mu.grid, otype_cartesian)
        right = [g.lattice(U_mu.grid, otype_cartesian) for _ in range(kernels.ng)]
        return kernels.combine(right, K)

    def _newton(self, fields, U_prime_mu, C, xparams):
        # site-local Newton steps for F(U_mu) = U_mu' (F the local map at the
        # fixed staple C, see g.algorithms.nonlinear.fixed_point.newton): the
        # residual r = TA(U_mu' F(U_mu)^dag) (the log up to third order), the
        # Jacobian block M (see jacobian_matrix) and U_mu <- exp(d) U_mu with
        # d = M^-1 c in the coordinates c of r (converges quadratically)
        TA = g.qcd.gauge.project.traceless_anti_hermitian

        def residual(x):
            return [TA(g(U_prime_mu * g.adj(self._local_ft(x[0], C, xparams))))]

        def solve(x, r):
            (U_mu,), (r,) = x, r
            kernels = g.group.algebra_kernels(U_mu.grid, U_mu.otype.cartesian())
            M, _ = self._local_jacobian_matrix(_with(fields, self.mu, U_mu), C)
            # (combine contracts the second index: transposed for d = M^-1 c;
            # right_a = -sum_b M^-1[b, a] T_b, so d = -sum_a c_a right_a)
            right = self._inverse_directions(M, U_mu, transpose=True)
            del M
            d = g.lattice(r)
            d[:] = 0
            for T, ra in zip(kernels.fgenerators, right):
                d -= g(g.trace(r * T) * (1.0 / kernels.norm)) * ra
            return [d]

        return g.algorithms.nonlinear.fixed_point.newton(
            residual, solve, compose=lambda d, U: g.matrix.exp(d) * U
        )

    def _project(self, sm, xparams):
        # TA(P1 f(sm, params)), the generator of the update; every path (the
        # graph of ft, the site-local map and inv) builds its update here
        if self.loop_function is not None:
            sm = g(self.loop_function(sm, xparams))
        if self.P1 is not None:
            sm = g(sm * self.P1)
        return g.qcd.gauge.project.traceless_anti_hermitian(sm)

    def log_det_jacobian_field(self, fields, staple=None):
        # the site-local log det of the mu->mu block (zero outside P1)
        return _masked(self.P1, g.component.log(g.matrix.det(self.jacobian_matrix(fields, staple))))

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
        # derivative, so the result is correctly differentiable.  (A second
        # reverse pass directly on the "apply Jacobian to a direction" graph
        # would differentiate a U-dependence that is not the correctly
        # trivialized one.)
        #
        # The nested reverse applies the infinitesimal_to_cartesian symmetrization
        # 0.5*(x + adj(x)) at the group leaf, which halves the derivative of the
        # 1st-derivative graph; the factor of 2 below compensates for it.
        rad = g.ad.reverse

        mu = self.mu
        N = len(fields)
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

        # resolved to plain values, so that the (expensive) nested node graphs
        # are released before the caller accumulates over the generators; a
        # field without gradient (a link not in the paths) gets a zero
        out = []
        for x, f in zip(grads, fields):
            r = resolve(x)
            out.append(g(2.0 * (g.group.zero(f) if r is None else r)))
        return out

    def action_log_det_jacobian_gradient(self, fields, dfields):
        if self.description_staple is not None:
            return self._local_action_log_det_jacobian_gradient(fields, dfields)
        return self._action_log_det_jacobian_gradient_generic(fields, dfields)

    def weighted_log_det_jacobian_gradient(self, fields, dfields, weight, staple=None):
        # the gradient of -sum_x weight(x) log det J(x) (weight: a real
        # complex field, fixed; staple(fields): a staple other than the
        # transport's own, e.g. prescribed sample staples depending on the
        # weights' parameters); local transports only
        assert self.description_staple is not None
        return self._local_action_log_det_jacobian_gradient(fields, dfields, weight, staple)

    def _local_action_log_det_jacobian_gradient(self, fields, dfields, weight=None, staple=None):
        # Same contraction as the generic version below, but the Jacobian
        # block is a site-local function of (U_mu, C): the 8 second-order
        # passes run over the stencil-free local map f(U_mu, C) and yield
        # gradients w.r.t. U_mu and C.  The C gradient is linear, so it is
        # summed over the generators first and pushed through the staple
        # stencil by ONE first-order reverse pass.
        rad = g.ad.reverse
        mu, P1 = self.mu, self.P1
        M, C = self._local_jacobian_matrix(fields, None if staple is None else g(staple(fields)))

        U_mu = fields[mu]
        generators = g.group.algebra_kernels(U_mu.grid, U_mu.otype.cartesian()).fgenerators
        ng = len(generators)
        right = self._inverse_directions(M, U_mu, weight=weight)
        del M
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
            grad_U = _add(grad_U, resolve(_U.gradient))
            grad_C = _add(grad_C, resolve(_C.gradient))
            for i, n in enumerate(_P):
                if n.gradient is not None:
                    grad_P[i] = _add(grad_P[i], resolve(n.gradient))

            del act, _left, _right
            right[a] = None

        del aU, aUft, _Up, _U, _C, _P
        del left, right, C, P1_node

        # chain rule through the staple; the factor 2 is the one explained in
        # diagonal_jacobian_gradient
        nodes = [rad.node(x) for x in fields]
        seed = g(2.0 * grad_C)
        del grad_C
        (self._staple if staple is None else staple)(nodes)(initial_gradient=seed)
        del seed
        out = []
        for i, n in enumerate(nodes):
            # (a field the staple does not depend on, e.g. with a prescribed
            # staple, gets a zero)
            r = g.group.zero(fields[i]) if n.gradient is None else g(n.gradient)
            if i == mu:
                r = g(r + 2.0 * grad_U)
            elif i >= self.nd and grad_P[i - self.nd] is not None:
                r = g(r + 2.0 * grad_P[i - self.nd])
            out.append(r)
        return [out[g.util.index_by_identity(fields, d)] for d in dfields]

    def _action_log_det_jacobian_gradient_generic(self, fields, dfields):
        # The action is -log det M (see jacobian_matrix), so
        #   \partial_rho (-log det M) = -tr(M^{-1} dM/drho)
        #                              = -sum_{a,b} M^{-1}[a,b] (dM/drho)[a,b].
        # Each term is \partial_rho <left, M.right> with left = P1*gen[a] and
        # right = -sum_b M^{-1}[a,b] gen[b] (the P1-masked site-dependent vector),
        # evaluated by diagonal_jacobian_gradient.
        M, fields_prime = self._jacobian_matrix_generic(fields)
        U_mu = fields[self.mu]
        right = self._inverse_directions(M, U_mu)
        del M
        left = g.group.cartesian(U_mu)
        gr_sum = None
        for T, ra in zip(g.group.algebra_kernels(U_mu.grid, U_mu.otype.cartesian()).fgenerators, right):
            left @= self.P1 * T
            gr = self.diagonal_jacobian_gradient(fields, fields_prime, left, ra)
            gr_sum = gr if gr_sum is None else [g(x + y) for x, y in zip(gr_sum, gr)]
        return [gr_sum[g.util.index_by_identity(fields, d)] for d in dfields]


class dpt_action_log_det_jacobian(differentiable_functional):
    def __init__(self, parent):
        self.parent = parent

    def __call__(self, fields):
        return -self.parent.log_det_jacobian(fields).real

    def gradient(self, fields, dfields):
        return self.parent.action_log_det_jacobian_gradient(fields, dfields)
