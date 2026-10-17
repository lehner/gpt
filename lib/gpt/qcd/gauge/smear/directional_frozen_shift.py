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
from gpt.core.group import differentiable_functional


def _with(fields, i, x):
    # fields with element i replaced by x
    return [x if j == i else f for j, f in enumerate(fields)]


class directional_frozen_shift(dft_diffeomorphism):
    def __init__(self, U, mu, P1, generator, P0=None, parameters=[]):
        # U_mu'(x) = exp(Omega(x)) U_mu(x), Omega = P1 TA(G(F, xparams)), all
        # other links unchanged.  P1 is a 0/1 mask field of the updated sites
        # (a checkerboard, a color of a lattice coloring, ...).  The generator
        # G sees only the frozen links F: the links nu != mu and U_mu P0 (P0
        # default 1 - P1, P0 P1 = 0), so the Jacobian is block-triangular
        # with a unit diagonal (log det = 0) and the inverse is explicit,
        # U_mu = exp(-Omega) U_mu' (F is not changed by the step), for any G.
        #
        # generator(F, xparams): a gauge-covariant N x N matrix field (M(x) ->
        # V(x) M(x) V(x)^dag), F the list of the nd frozen links, xparams the
        # parameters (plain values or nodes, in the order of `parameters`).
        # It must work on plain fields and on nodes (the force is its reverse
        # AD) and must not read U_mu on P1, where F has zeros (which breaks
        # gauge covariance; see check_frozen).
        nd = len(U)
        self.nd = nd
        self.mu = mu
        self.P1 = P1
        if P0 is None:
            P0 = g.lattice(P1)
            P0[:] = 1
            P0 = g(P0 - P1)
        self.P0 = P0
        if g.norm2(P0 * P1) != 0.0:
            raise ValueError("directional_frozen_shift: P0 P1 != 0 (the generator would read updated links)")
        self.generator = generator
        self.parameters = parameters
        # every output but U_mu is the identity map of its input (see
        # g.group.transformed: jacobian accepts the needed inputs)
        self.identity_outputs = [i for i in range(nd + len(parameters)) if i != mu]

        def ft(xfields):
            assert len(xfields) == nd + len(parameters)
            return _with(xfields, mu, self._update(xfields, xfields[mu], -1.0))

        super().__init__(U + parameters, ft)

    def _frozen(self, xfields):
        # the frozen links F (plain fields or nodes)
        mu = self.mu
        return [g(x * self.P0) if nu == mu else x for nu, x in enumerate(xfields[0 : self.nd])]

    def generator_field(self, xfields):
        # Omega = P1 TA(G(F, xparams))
        M = self.generator(self._frozen(xfields), xfields[self.nd :])
        return g.qcd.gauge.project.traceless_anti_hermitian(g(M * self.P1))

    def _update(self, xfields, xU_mu, sign):
        # exp(-sign Omega) xU_mu, Omega from the frozen links of xfields
        Omega = self.generator_field(xfields)
        return g(g.matrix.exp(g((-sign) * Omega)) * xU_mu)

    def inv(self, fields, max_iter=None, eps=None):
        # exact: the frozen links of the image are those of the preimage
        # (max_iter and eps for the interface of directional_parallel_transport)
        return _with(fields, self.mu, self._update(fields, fields[self.mu], +1.0))

    def jacobian(self, fields, fields_prime, dfields, inputs=None):
        # only output mu is transformed, every other output is the identity
        # map of its input: one reverse pass from output mu (into U_mu itself
        # and, through the generator, into the frozen links and parameters),
        # plus the identity parts.  inputs: the positions of the fields whose
        # gradient is needed (None: all)
        mu, N = self.mu, len(fields)
        assert len(fields_prime) == N and len(dfields) == N
        needed = list(range(N)) if inputs is None else sorted(set(inputs) | {mu})
        seed = g.cartesian_to_infinitesimal(fields_prime[mu], dfields[mu])
        self._set_leaves(fields)
        for leaf in self.aU:
            leaf.gradient = None
        self.aUft[mu].backward(initial_gradient=seed, wrt=[self.aU[i] for i in needed])
        gradient = []
        for nu in range(N):
            gr = self.aU[nu].gradient if nu in needed else None
            if nu != mu:
                if gr is None:
                    gr = g.copy(dfields[nu]) if isinstance(dfields[nu], g.lattice) else dfields[nu]
                else:
                    gr = g(gr + dfields[nu]) if isinstance(gr, g.lattice) else gr + dfields[nu]
            elif gr is None:
                gr = g(0 * dfields[nu])
            if isinstance(gr, g.lattice):
                gr.otype = dfields[nu].otype
            gradient.append(gr)
        # (the cached leaves keep no gradient until the next call: the returned
        # list holds its own references; else four fields per step stay alive)
        for leaf in self.aU:
            leaf.gradient = None
        return gradient

    def log_det_jacobian_field(self, fields):
        z = g.complex(self.P1.grid)
        z[:] = 0
        return z

    def log_det_jacobian(self, fields):
        return 0.0

    def action_log_det_jacobian(self):
        return zero_functional()

    def check_frozen(self, fields, rng=None, eps=None):
        # the generator reads no updated link: Omega on P1 is the same for the
        # frozen view (zeros at U_mu on P1) and for the frozen view with random
        # links at U_mu on P1 (any read of these sites changes it; reading the
        # zeros instead would not break the exact inverse or log det = 0, but
        # gauge covariance and smoothness).  Raises otherwise; returns the
        # relative change.  (Not a nan fill: it would flag generators that
        # read a link and remove it with a mask product, nan * 0 = nan, which
        # are exactly frozen.)
        mu, nd, P1 = self.mu, self.nd, self.P1
        zero = g.lattice(fields[mu].grid, fields[mu].otype.cartesian())
        zero[:] = 0

        def generator_on_P1(F):
            M = self.generator(F, fields[nd:])
            return g.where(P1, g.qcd.gauge.project.traceless_anti_hermitian(M), zero)

        if rng is None:
            rng = g.random("directional_frozen_shift.check_frozen")
        other = g.lattice(fields[mu])
        rng.element(other)
        F = self._frozen(fields)
        F_random = _with(F, mu, g(F[mu] + other * P1))
        O0 = generator_on_P1(F)
        O1 = generator_on_P1(F_random)
        n0 = g.norm2(O0)
        change = (g.norm2(O1 - O0) / (n0 if n0 > 0 else 1.0)) ** 0.5
        if eps is None:
            eps = 100 * fields[mu].grid.precision.eps
        if not change <= eps:
            raise ValueError(f"directional_frozen_shift: the generator reads U_{mu} on the updated sites ({change})")
        return change


class zero_functional(differentiable_functional):
    # the identically vanishing action log det of a volume-preserving step:
    # transformed by any map it stays zero (no reverse pass), and zero + x = x
    def __init__(self):
        pass

    def __call__(self, fields):
        return 0.0

    def gradient(self, fields, dfields):
        return [g.group.zero(d) for d in g.util.to_list(dfields)]

    def transformed(self, t, indices=None, projection=None):
        return self

    def __add__(self, other):
        return other

    def __radd__(self, other):
        assert other == 0 or isinstance(other, zero_functional)
        return self
