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
from gpt.core.parallel_transport.matrix import new_target_list


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


def _zero_like(x):
    z = g.lattice(x)
    z[:] = 0
    return z


def _reset_gradients(nodes):
    # a reverse pass resets the gradients of the leaves it reaches only: a
    # leaf outside its graph would keep the gradient of an earlier pass
    for x in nodes:
        x.gradient = None


def _link_offsets(path, nd, mu):
    # the sites x + s of the links U_mu(x + s) a path from x traverses (in
    # either direction), and whether the path is closed
    pos = [0] * nd
    offsets = []
    for nu, d in path.path:
        for _ in range(abs(d)):
            if d < 0:
                pos[nu] -= 1
            if nu == mu:
                offsets.append(tuple(pos))
            if d > 0:
                pos[nu] += 1
    return offsets, all(x == 0 for x in pos)


def _shifted(x, offset):
    # y(z) = x(z + offset)
    for d, o in enumerate(offset):
        if o != 0:
            x = g.cshift(x, d, o)
    return x


class directional_parallel_transport(dft_diffeomorphism):
    def __init__(
        self,
        U,
        description_mu,
        mu,
        P0=None,
        P1=None,
        parameters=[],
        loop_function=None,
        loops=None,
    ):
        # loop_function: an optional site-local map f(sm, xparams) of the
        # weighted loop sum, U_mu' = exp(TA(P1 f(sm, xparams))) U_mu (default:
        # the identity), where xparams are the parameters (plain fields or
        # nodes, in the order of `parameters`).  It must work on plain fields
        # and on nodes (node-first products: sm * p, not p * sm)
        # and be gauge covariant (products of sm and adj(sm), traces).
        #
        # loops: an optional (non-empty) list of closed g.path loops at x,
        # transported with the links (unweighted, P0 applied to U_mu).  The loop
        # function is then f(sm, xparams, L) with L the list of the loops
        # L_k(x) (N x N fields, L_k -> V(x) L_k V(x)^dag).  A loop must not
        # traverse a link the step updates (U_mu on the sites of P1) when
        # evaluated at a site of P1: it is then fixed during the step like the
        # staple, enters the local map as a constant and is evaluated once by
        # inv.  This is checked here (ValueError otherwise), e.g. the 2x1
        # rectangle around U_mu(x) (the two plaquettes sharing it) with a
        # checkerboard P1.
        self.description_mu = description_mu
        self.mu = mu
        self.P0 = P0
        self.P1 = P1
        self.parameters = parameters
        self.loop_function = loop_function

        nd = len(U)
        self.nd = nd
        fields = U + parameters

        self.loops = None if loops is None else list(loops)
        self._loop_cache = {}
        if self.loops is not None:
            self._check_loops()

        cache = {}

        def ft(xU):
            assert len(xU) == nd + len(parameters)
            sm = self._weighted_transport(cache, description_mu, xU)
            return _with(xU, mu, self._update(sm, xU[mu], xU[nd:], self._loops(xU)))

        self.description_staple = g.staple_description(description_mu, mu, nd)
        self._staple_cache = {}
        # the cached VJP graphs, with and without gradient-carrying parameter
        # leaves
        self._vjp = {}
        # every output but U_mu is the identity map of its input (see
        # g.group.transformed: their gradients are needed only when requested,
        # and jacobian accepts the needed inputs)
        self.identity_outputs = [i for i in range(nd + len(parameters)) if i != mu]

        super().__init__(fields, ft)

    def _weighted_transport(self, cache, description, xU):
        # sum_k weight_k * transport_k(U), with P0 applied to U_mu; works for
        # plain fields and for nodes
        nd, mu, P0, parameters = self.nd, self.mu, self.P0, self.parameters

        cache_key = str(type(xU[0]))
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

    def _check_loops(self):
        # a loop evaluated at x in P1 must not traverse U_mu(x + s) with
        # x + s in P1: P1(x) P1(x + s) = 0 for every such offset s
        if len(self.loops) == 0:
            raise ValueError("directional_parallel_transport: loops must be None or non-empty")
        overlap = {}
        for k, p in enumerate(self.loops):
            offsets, closed = _link_offsets(p, self.nd, self.mu)
            if not closed:
                raise ValueError(f"directional_parallel_transport: loop {k} is not closed")
            for s in offsets:
                if s not in overlap:
                    overlap[s] = self.P1 is None or g.norm2(self.P1 * _shifted(self.P1, s)) > 0
                if overlap[s]:
                    raise ValueError(
                        f"directional_parallel_transport: loop {k} traverses the link "
                        f"U_{self.mu}(x + {s}), which the step updates for some x in P1"
                    )

    def _loop_transport(self, xfields):
        # the loops L_k of the links in xfields (plain fields or nodes) in one
        # stencil; returns (root, L), root the stencil's output
        # node (for nodes, a list node for several loops, None for plain
        # fields with several loops), whose reverse pass takes one flow per
        # loop
        nd, K, mu, P0 = self.nd, len(self.loops), self.mu, self.P0
        links = xfields[0:nd]
        key = str(type(links[0]))
        # (P0 applied to U_mu, as for the staple)
        if P0 is not None:
            links = [g(links[i] * P0) if i == mu else links[i] for i in range(nd)]
        if key not in self._loop_cache:
            code = [(k, -1, 1.0, p) for k, p in enumerate(self.loops)]
            self._loop_cache[key] = g.parallel_transport_matrix(links, code, K)
        ptm = self._loop_cache[key]
        if K > 1 and isinstance(links[0], g.ad.reverse.node_base):
            # (keep the list node: parallel_transport_matrix returns its elements)
            T = new_target_list(links[0], K)
            ptm.stencil(T, *links)
            return T, [T[k] for k in range(K)]
        L = g.util.to_list(ptm(links))
        return (L[0] if K == 1 else None), L

    def _loops(self, xfields):
        # the list of loops (None without loops)
        return None if self.loops is None else self._loop_transport(xfields)[1]

    def _loop_vjp(self, fields, flows, loops=None):
        # the gradients w.r.t. the fields of the loops with the flows (one
        # per loop, None: zero): one reverse pass of the loops' stencil, or of
        # prescribed loops(fields), one pass per loop; None for a field that
        # receives nothing
        rad = g.ad.reverse
        nodes = [rad.node(x) for x in fields]
        grads = [None] * len(fields)
        if all(x is None for x in flows):
            return grads
        if loops is None:
            root, _ = self._loop_transport(nodes)
            template = next(x for x in flows if x is not None)
            flows = [_zero_like(template) if x is None else x for x in flows]
            passes = [(root, flows if len(flows) > 1 else flows[0])]
        else:
            passes = [
                (L_k, flow)
                for L_k, flow in zip(loops(nodes), flows)
                if flow is not None and isinstance(L_k, rad.node_base)
            ]
        for root, flow in passes:
            _reset_gradients(nodes)
            # (prescribed loops: the passes share one forward)
            root.backward(initial_gradient=flow, retain_values=loops is not None)
            for i, x in enumerate(nodes):
                if x.gradient is not None:
                    grads[i] = _add(grads[i], x.gradient)
        return grads

    def _update(self, sm, xU_mu, xparams, xloops=None):
        # U_mu' = exp(TA(P1 f(sm))) U_mu
        return g(g.matrix.exp(self._project(sm, xparams, xloops)) * xU_mu)

    def _staple(self, xfields):
        # the weighted staple C with transported loop = C U_mu^dag (P0 applied)
        return g(self._weighted_transport(self._staple_cache, self.description_staple, xfields))

    def _local_generator(self, xU_mu, xC, xparams, xloops=None):
        # TA(P1 f(C U_mu^dag)) for a fixed staple C (P0 applied to U_mu) and
        # fixed loops
        xU_P0 = g(xU_mu * self.P0) if self.P0 is not None else xU_mu
        return self._project(g(xC * g.adj(xU_P0)), xparams, xloops)

    def _local_ft(self, xU_mu, xC, xparams, xloops=None):
        # the site-local map U_mu' = F(U_mu, C, params, L) for a fixed staple
        # C and fixed loops L
        return g(g.matrix.exp(self._local_generator(xU_mu, xC, xparams, xloops)) * xU_mu)

    def jacobian(self, fields, fields_prime, dfields, inputs=None):
        # only output mu is transformed, every other output is the identity
        # map, whose Jacobian passes its direction through: one reverse pass
        # (for output mu) instead of one per output.  inputs: the positions of
        # the fields whose gradient is needed (None: all); the parameters'
        # flows are computed only if one of them is needed (else their
        # gradients are only the identity part)
        mu = self.mu
        N = len(fields)
        assert len(fields_prime) == N and len(dfields) == N
        seed = g.cartesian_to_infinitesimal(fields_prime[mu], dfields[mu])
        if self.description_staple is not None:
            params = inputs is None or any(i >= self.nd for i in inputs)
            grads = self._local_vjp(fields, seed, params)
        else:
            self._set_leaves(fields)
            for leaf in self.aU:
                leaf.zero_gradient()
            self.aUft[mu].backward(initial_gradient=seed)
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

    def _local_vjp(self, fields, seed, params=True):
        # the VJP of output mu through the site-local map f(U_mu, C, L), the
        # staple C (see g.staple_description) and the loops L: the staple
        # stencil and its adjoint carry one link fewer per path than the
        # transported loop
        rad = g.ad.reverse
        mu = self.mu
        # the graphs are built once and re-used with new leaf values: node
        # graphs are reference cycles, so graphs built per call would only be
        # released by the cyclic garbage collector, which does not see the
        # size of the fields they hold
        plain = not any(isinstance(x, rad.node_base) for x in fields)
        nd = self.nd
        if plain and params in self._vjp:
            nodes, aC, aL, _U, _C, _L, _P, aF = self._vjp[params]
            for n, x in zip(nodes + _P, fields + fields[nd:]):
                assert_compatible(n.value, x)
                n.value = x
        else:
            nodes = [rad.node(x) for x in fields]
            aC = self._staple(nodes)
            _U = rad.node(fields[mu])
            # a plain matrix (not a group element): no conversion to the algebra
            _C = rad.node(g.lattice(fields[mu]), infinitesimal_to_cartesian=False)
            # the loops enter like the staple (aL: the root of their stencil)
            aL, _L = None, []
            if self.loops is not None:
                aL, _ = self._loop_transport(nodes)
                _L = [
                    rad.node(g.lattice(fields[mu]), infinitesimal_to_cartesian=False)
                    for _ in self.loops
                ]
            # separate parameter leaves for the local map (the staple pass
            # below would reset gradients deposited in shared leaves)
            _P = [rad.node(x, with_gradient=params) for x in fields[nd:]]
            aF = self._local_ft(_U, _C, _P, _L if aL is not None else None)
            if plain:
                self._vjp[params] = (nodes, aC, aL, _U, _C, _L, _P, aF)
        roots = [aC, aF] + ([] if aL is None else [aL])
        # (values a previous call retained, see the release below)
        for r in roots:
            r.value = None
        # one staple (loops) forward, shared by the local pass and the
        # staple's (loops') reverse
        _C.value = aC(with_gradients=False, retain_values=True)
        if aL is not None:
            for n, x in zip(_L, g.util.to_list(aL(with_gradients=False, retain_values=True))):
                n.value = x
        _U.value = fields[mu]
        aF.backward(initial_gradient=seed)
        grad_U, grad_C = _U.gradient, _C.gradient
        grad_P = [n.gradient for n in _P]
        grad_L = [n.gradient for n in _L]
        aC.backward(initial_gradient=grad_C)
        grads = [n.gradient for n in nodes]
        if any(x is not None for x in grad_L):
            flows = [_zero_like(_L[0].value) if x is None else x for x in grad_L]
            _reset_gradients(nodes)
            aL.backward(initial_gradient=flows if len(flows) > 1 else flows[0])
            for i, n in enumerate(nodes):
                if n.gradient is not None:
                    grads[i] = _add(grads[i], n.gradient)
        for i, gr in [(mu, grad_U)] + [(nd + i, gr) for i, gr in enumerate(grad_P)]:
            if gr is not None:
                grads[i] = _add(grads[i], gr)
        # release the fields held by the graph until the next call
        for r in roots:
            r.value = None
        for n in [_C] + _L:
            n.value = None
        for n in nodes + [_U, _C] + _L + _P:
            n.gradient = None
        return grads

    def jacobian_matrix(self, fields, staple=None, loops=None):
        # the mu->mu block M[a, b] = coordinate b of the VJP of output mu with
        # the generator a (masked to P1); staple(fields): a staple other than
        # the transport's own (local transports only), e.g. prescribed sample
        # staples; loops(fields) likewise: a list of loops other than the
        # transport's own
        if self.description_staple is not None:
            return self._local_jacobian_matrix(fields, *self._prescribed(fields, staple, loops))[0]
        assert staple is None and loops is None
        return self._jacobian_matrix_generic(fields)[0]

    def _prescribed(self, fields, staple, loops):
        # the plain staple and loops of prescribed functions (None: the own)
        assert loops is None or self.loops is not None
        C = None if staple is None else g(staple(fields))
        L = None if loops is None else [g(x) for x in loops(fields)]
        assert L is None or len(L) == len(self.loops)
        return C, L

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

    def _local_jacobian_matrix(self, fields, C=None, L=None):
        # the block from VJPs through the stencil-free local map f(U_mu, C, L)
        # at fixed staple C (default: the staple of fields, see
        # g.staple_description) and fixed loops L (default: the loops of
        # fields), one forward shared by the passes; returns the block, the
        # staple and the loops
        rad = g.ad.reverse
        if C is None:
            C = self._staple(fields)
        if L is None:
            L = self._loops(fields)
        U_mu = fields[self.mu]
        aU = rad.node(U_mu)
        xparams = [rad.node(x, with_gradient=False) for x in fields[self.nd :]]
        xloops = None if L is None else [rad.node(x, with_gradient=False) for x in L]
        aUft = self._local_ft(aU, rad.node(C, with_gradient=False), xparams, xloops)
        U_prime_mu = aUft(with_gradients=False, retain_values=True)

        def vjp(seed):
            aUft(initial_gradient=seed, retain_values=True)
            return aU.gradient

        return self._block(U_mu, U_prime_mu, vjp), C, L

    def _jacobian_matrix_generic(self, fields):
        # the block from VJPs through the full transport graph; returns the
        # block and the transported fields
        mu = self.mu
        fields_prime = self(fields)
        self._set_leaves(fields)

        def vjp(seed):
            self.aUft[mu].backward(initial_gradient=seed)
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
        # (the loops likewise: fixed during the step, verified below)
        C = self._staple(fields)
        L = self._loops(fields)
        xparams = fields[nd:]
        U_prime_mu = fields[mu]
        precision = U_prime_mu.grid.precision.eps

        def step(x):
            x[0] @= g.matrix.exp(g(-1.0 * self._local_generator(x[0], C, xparams, L))) * U_prime_mu

        newton_step = self._newton(fields, U_prime_mu, C, xparams, L)
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
            M = self._local_jacobian_matrix(U, C, L)[0]
            det = _masked(self.P1, g.matrix.det(M), 1.0)[:].real
            del M
            if det.min() <= 0:
                raise RuntimeError(
                    f"directional_parallel_transport could not be inverted: the Newton solution has "
                    f"det M <= 0 at {int((det <= 0).sum())} sites (the site map is not a bijection)"
                )
        # the staple of the result must be the one used above (where it
        # enters the update)
        checks = [("staple", self._staple(U), C)]
        if L is not None:
            checks += [(f"loop {k}", a, b) for k, (a, b) in enumerate(zip(self._loops(U), L))]
        for name, a, b in checks:
            eps2 = g.norm2(self.P1 * (a - b)) / max(g.norm2(self.P1 * b), 1e-300)
            if not eps2 < 1e4 * precision**2:
                raise RuntimeError(
                    f"directional_parallel_transport: the {name} depends on the updated links ({eps2})"
                )
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

    def _newton(self, fields, U_prime_mu, C, xparams, L=None):
        # site-local Newton steps for F(U_mu) = U_mu' (F the local map at the
        # fixed staple C and loops L, see g.algorithms.nonlinear.fixed_point.newton): the
        # residual r = TA(U_mu' F(U_mu)^dag) (the log up to third order), the
        # Jacobian block M (see jacobian_matrix) and U_mu <- exp(d) U_mu with
        # d = M^-1 c in the coordinates c of r (converges quadratically)
        TA = g.qcd.gauge.project.traceless_anti_hermitian

        def residual(x):
            return [TA(g(U_prime_mu * g.adj(self._local_ft(x[0], C, xparams, L))))]

        def solve(x, r):
            (U_mu,), (r,) = x, r
            kernels = g.group.algebra_kernels(U_mu.grid, U_mu.otype.cartesian())
            M = self._local_jacobian_matrix(_with(fields, self.mu, U_mu), C, L)[0]
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

    def _project(self, sm, xparams, xloops=None):
        # TA(P1 f(sm, params[, loops])), the generator of the update; every
        # path (the graph of ft, the site-local map and inv) builds its update
        # here
        if self.loop_function is not None:
            if self.loops is None:
                sm = g(self.loop_function(sm, xparams))
            else:
                sm = g(self.loop_function(sm, xparams, xloops))
        if self.P1 is not None:
            sm = g(sm * self.P1)
        return g.qcd.gauge.project.traceless_anti_hermitian(sm)

    def log_det_jacobian_field(self, fields, staple=None, loops=None):
        # the site-local log det of the mu->mu block (zero outside P1)
        M = self.jacobian_matrix(fields, staple, loops)
        return _masked(self.P1, g.component.log(g.matrix.det(M)))

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
        # Build left . (\partial f_mu/\partial U_mu) . right as a functional
        # over (fields, left, right) and take its gradient.  The mu->mu block
        # is the "apply Jacobian to a direction" reverse pass w.r.t. U_mu,
        # recorded (create_graph), with a seed built on the transport's own
        # node; the recorded gradient is a graph over the same leaves, whose
        # reverse pass is a true 1st derivative, so the result is correctly
        # differentiable.  (A second reverse pass directly on the "apply
        # Jacobian to a direction" graph would differentiate a U-dependence
        # that is not the correctly trivialized one.)
        #
        # The recorded reverse applies the infinitesimal_to_cartesian
        # symmetrization 0.5*(x + adj(x)) at the group leaf, which halves the
        # derivative of the 1st-derivative graph; the factor of 2 below
        # compensates for it.
        rad = g.ad.reverse

        mu = self.mu
        _U = [rad.node(g.copy(u)) for u in fields]
        _left = rad.node(g.copy(left), with_gradient=False)
        _right = rad.node(g.copy(right), with_gradient=False)

        _Up = self.ft(_U)
        seed = g.cartesian_to_infinitesimal(_Up[mu], _right)
        _Up[mu].backward(initial_gradient=seed, create_graph=True, wrt=[_U[mu]])
        J_right_mu = _U[mu].gradient             # graph = (\partial f_mu/\partial U_mu) . right

        act = g.inner_product(_left, rad.node(self.P1, with_gradient=False) * J_right_mu)
        func = act.functional(*(_U + [_left, _right]))
        grads = func.gradient(fields + [left, right], fields)   # gradient w.r.t. _U

        # (a field without gradient, a link not in the paths, gets a zero)
        out = []
        for x, f in zip(grads, fields):
            out.append(g(2.0 * (g.group.zero(f) if x is None else x)))
        return out

    def action_log_det_jacobian_gradient(self, fields, dfields):
        if self.description_staple is not None:
            return self._local_action_log_det_jacobian_gradient(fields, dfields)
        return self._action_log_det_jacobian_gradient_generic(fields, dfields)

    def weighted_log_det_jacobian_gradient(self, fields, dfields, weight, staple=None, loops=None):
        # the gradient of -sum_x weight(x) log det J(x) (weight: a real
        # complex field, fixed; staple(fields): a staple other than the
        # transport's own, e.g. prescribed sample staples depending on the
        # weights' parameters; loops(fields) likewise); local transports only
        assert self.description_staple is not None
        return self._local_action_log_det_jacobian_gradient(fields, dfields, weight, staple, loops)

    def _local_action_log_det_jacobian_gradient(
        self, fields, dfields, weight=None, staple=None, loops=None
    ):
        # Same contraction as the generic version below, but the Jacobian
        # block is a site-local function of (U_mu, C, L): the 8 second-order
        # passes run over the stencil-free local map f(U_mu, C, L) and yield
        # gradients w.r.t. U_mu, C and the loops L.  The C (L) gradient is
        # linear, so it is summed over the generators first and pushed through
        # the staple (loops) stencil by ONE first-order reverse pass.
        rad = g.ad.reverse
        mu, P1 = self.mu, self.P1
        M, C, L = self._local_jacobian_matrix(fields, *self._prescribed(fields, staple, loops))

        U_mu = fields[mu]
        generators = g.group.algebra_kernels(U_mu.grid, U_mu.otype.cartesian()).fgenerators
        ng = len(generators)
        right = self._inverse_directions(M, U_mu, weight=weight)
        del M
        left = g.group.cartesian(U_mu)
        P1_node = rad.node(P1, with_gradient=False)

        # leaves (never modified, no copies needed); the staple leaf is a
        # plain matrix (not a group element), so its gradient is not
        # converted to the algebra
        _U = rad.node(U_mu)
        _C = rad.node(C, infinitesimal_to_cartesian=False)
        # (parameter flows only if a parameter is requested)
        params = any(g.util.index_by_identity(fields, d) >= self.nd for d in dfields)
        _P = [rad.node(x, with_gradient=params) for x in fields[self.nd :]]
        _L = None if L is None else [rad.node(x, infinitesimal_to_cartesian=False) for x in L]

        # "apply Jacobian block to right" recorded w.r.t. U_mu (see
        # diagonal_jacobian_gradient); the forward does not depend on the
        # generator, so it runs once and its values are shared by all
        # passes.  The seed is built on the transport's own node, so the
        # derivative graph and the seed share its nodes.
        aUft = self._local_ft(_U, _C, _P, _L)
        aUft(with_gradients=False, retain_values=True)

        grad_U = None
        grad_C = None
        grad_P = [None] * len(_P)
        grad_L = [None] * (0 if _L is None else len(_L))
        for a in range(ng):
            left @= P1 * generators[a]
            _left = rad.node(left, with_gradient=False)
            _right = rad.node(right[a], with_gradient=False)

            # both reverse passes retain the shared forward values; the
            # per-generator part of the graph is released with act
            aUft.backward(
                initial_gradient=g.cartesian_to_infinitesimal(aUft, _right),
                retain_values=True,
                create_graph=True,
                wrt=[_U],
            )
            act = g.inner_product(_left, P1_node * _U.gradient)
            act.backward(retain_values=True)
            grad_U = _add(grad_U, _U.gradient)
            grad_C = _add(grad_C, _C.gradient)
            for i, n in enumerate(_P + (_L or [])):
                if n.gradient is not None:
                    if i < len(_P):
                        grad_P[i] = _add(grad_P[i], n.gradient)
                    else:
                        grad_L[i - len(_P)] = _add(grad_L[i - len(_P)], n.gradient)

            del act, _left, _right
            right[a] = None

        del aUft, _U, _C, _P, _L
        del left, right, C, L, P1_node

        # chain rule through the staple and the loops; the factor 2 is the
        # one explained in diagonal_jacobian_gradient
        nodes = [rad.node(x) for x in fields]
        seed = g(2.0 * grad_C)
        del grad_C
        (self._staple if staple is None else staple)(nodes).backward(initial_gradient=seed)
        del seed
        grads = [n.gradient for n in nodes]
        if self.loops is not None:
            flows = [None if x is None else g(2.0 * x) for x in grad_L]
            del grad_L
            for i, gr in enumerate(self._loop_vjp(fields, flows, loops)):
                if gr is not None:
                    grads[i] = _add(grads[i], gr)
            del flows
        out = []
        for i, gr in enumerate(grads):
            # (a field the staple does not depend on, e.g. with a prescribed
            # staple, gets a zero)
            r = g.group.zero(fields[i]) if gr is None else g(gr)
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
