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
from gpt.core.group import local_diffeomorphism, differentiable_functional


def _number(x):
    # a number parameter stored as a 0-d array enters as its element (plain
    # or node); fields and numbers as they are
    if isinstance(x, np.ndarray):
        return x[()]
    if isinstance(x, g.ad.reverse.node_base) and x._container.tag[0] is np.ndarray:
        return x[()]
    return x


class spectral_loop(local_diffeomorphism):
    # the spectral single-loop step on the links U_mu of one checkerboard P1:
    #
    #   U_mu'(x) = exp(TA(P1 F(P(x)))) U_mu(x),   P = T0 U_mu^dag,   F(P) = sum_k c_k P^k,
    #
    # T0 the staple x -> x + mu through x + nu0 (forward) or x - nu0
    # (backward), built from links the step does not change (the loop
    # g.path().f(nu0).f(mu).b(nu0).b(mu), resp. .b(nu0).f(mu).f(nu0).b(mu), as
    # a directional_parallel_transport with loop function F).  F commutes with
    # P, so the step moves only the eigenphases of P; it is a bijection if
    # sum_k k |c_k| < 1 at every site of P1.  SU(3) only: the log det is
    # analytic (Weyl integration formula),
    #
    #   log det J = P1 [ log(e2(1 - B)/3) + log(-Disc(P')) - log(-Disc(P)) ],
    #   B = (1/2) sum_k k (c_k P^k + h.c.),   e2(A) = ((tr A)^2 - tr A^2)/2,   P' = T0 U_mu'^dag,
    #   Disc(P) = |e1|^4 - 4 (e1^3 + conj(e1)^3) + 18 |e1|^2 - 27 <= 0,   e1 = tr P,
    #
    # a node graph of traces of matrix polynomials (any derivative order).
    #
    # coefficients: a dict {k: c_k} of numbers (fixed), or a set of degrees
    # {k, ...} (learnable): the fields are then U + [c_k for k in sorted
    # degrees], each c_k a number (or 0-d array) or a complex field (site
    # dependent), plain or a node (e.g. the output of another network).  The
    # bijectivity bound is checked on plain values (ValueError); for nodes
    # the computation providing them is responsible.
    def __init__(self, dimension, checkerboard, nu0, forward=True, coefficients={1: 0.12}):
        assert nu0 != dimension
        self.mu, self.checkerboard, self.nu0, self.forward = dimension, checkerboard, nu0, forward
        if isinstance(coefficients, dict):
            self.fixed = dict(coefficients)
            self.degrees = sorted(self.fixed.keys())
            self._check_bijective([self.fixed[k] for k in self.degrees])
        else:
            self.fixed = None
            self.degrees = sorted(coefficients)
        assert all(isinstance(k, int) and k >= 1 for k in self.degrees) and self.degrees
        # learnable coefficient fields
        self.n_parameters = 0 if self.fixed is not None else len(self.degrees)
        # the number of links (that of the last fields seen; 4 before)
        self.nd = 4
        self.cache = {}

    def _split(self, fields):
        nd = len(fields) - self.n_parameters
        assert nd > max(self.mu, self.nu0)
        return nd, list(fields[0:nd]), list(fields[nd:])

    def transport(self, fields):
        # the single-loop directional_parallel_transport with loop function F,
        # one per grid and gauge group, built at the first use
        nd, U, cs = self._split(fields)
        self.nd = nd
        key = (U[0].grid, U[0].otype.__name__)
        if key not in self.cache:
            assert U[0].otype.Nc == 3, "spectral_loop: the log det assumes SU(3)"
            mu, nu = self.mu, self.nu0
            P1 = self._P1(U[0].grid)
            if self.forward:
                path = g.path().f(nu).f(mu).b(nu).b(mu)
            else:
                path = g.path().b(nu).f(mu).f(nu).b(mu)
            one = g.complex(U[0].grid)
            one[:] = 1.0
            identity = g.identity(U[0])
            t = g.qcd.gauge.smear.directional_parallel_transport(
                U, [(1.0, path)], mu, None, P1, cs, loop_function=self._F
            )
            self.cache[key] = (t, P1, one, identity, {})
        return self.cache[key]

    def _coefficients(self, xparams):
        if self.fixed is not None:
            return [self.fixed[k] for k in self.degrees]
        return [_number(c) for c in xparams]

    def _powers(self, P):
        # P^k for k = 1..max degree (plain or nodes)
        powers = [P]
        for _ in range(self.degrees[-1] - 1):
            powers.append(g(powers[-1] * P))
        return powers

    def _F(self, P, xparams):
        # sum_k c_k P^k (node-first products: P^k c_k)
        powers = self._powers(P)
        F = None
        for k, c in zip(self.degrees, self._coefficients(xparams)):
            t = powers[k - 1] * c
            F = t if F is None else F + t
        return g(F)

    def _check_bijective(self, cs):
        # sum_k k |c_k| < 1 at every site of P1 (plain values; numbers and
        # fields)
        bound = 0.0
        field = None
        for k, c in zip(self.degrees, cs):
            if isinstance(c, g.lattice):
                t = g(k * g.component.abs(c))
                field = t if field is None else g(field + t)
            else:
                bound += k * abs(complex(np.asarray(c)))
        if field is None:
            if bound >= 1:
                raise ValueError(f"spectral_loop: sum_k k |c_k| = {bound} >= 1 (the step is not a bijection)")
            return
        v = g(self._P1(field.grid) * field)[:].real + bound
        violations = field.grid.globalsum(float((v >= 1).sum()))
        if violations > 0:
            raise ValueError(
                f"spectral_loop: sum_k k |c_k| >= 1 at {int(violations)} sites (the step is not a bijection)"
            )

    def _P1(self, grid):
        even, odd = g.even_odd_projectors(grid)
        return even if self.checkerboard is g.even else odd

    def _plain(self, fields):
        return not any(isinstance(x, g.ad.reverse.node_base) for x in fields)

    def __call__(self, fields):
        t = self.transport(fields)[0]
        if self.fixed is None and self._plain(fields):
            self._check_bijective(self._split(fields)[2])
        return t(fields)

    @property
    def identity_outputs(self):
        # every output but U_mu (the other links and the coefficient fields)
        # is the identity map of its input
        return [i for i in range(self.nd + self.n_parameters) if i != self.mu]

    def inv(self, fields, max_iter=100, eps=None):
        t = self.transport(fields)[0]
        if self.fixed is None and self._plain(fields):
            self._check_bijective(self._split(fields)[2])
        return t.inv(fields, max_iter=max_iter, eps=eps)

    def jacobian(self, fields, fields_prime, dfields, inputs=None):
        return self.transport(fields)[0].jacobian(fields, fields_prime, dfields, inputs)

    def _log_det(self, fields, C=None, U_prime_mu=None):
        # the site-local log det (plain fields or nodes), zero outside P1; C
        # and U_prime_mu: the staple and the updated link if already built
        # (shared with the graph of the map)
        t, P1, one, identity, _ = self.transport(fields)
        nd, U, cs = self._split(fields)
        mu = self.mu
        if C is None:
            C = t._staple(fields)
        if U_prime_mu is None:
            U_prime_mu = t._local_ft(U[mu], C, cs)
        P = g(C * g.adj(U[mu]))
        Pp = g(C * g.adj(U_prime_mu))
        # tr B and tr B^2 from the power sums s_n = tr P^n (Newton's
        # identities for SU(3): s_n = e1 s_{n-1} - conj(e1) s_{n-2} + s_{n-3},
        # s_0 = 3, s_{-n} = conj(s_n)), so no matrix B:
        #   tr B = -Re sum_k k c_k s_k,
        #   tr B^2 = (1/2) Re sum_kl k l (c_k c_l s_{k+l} + c_k conj(c_l) s_{k-l})
        conj = g.ad.reverse.transform.conj
        e1 = g(g.trace(P))
        e1c = conj(e1)
        s = [g(3.0 * one), e1]
        for n in range(2, 2 * self.degrees[-1] + 1):
            r = g(e1 * s[n - 1] - e1c * s[n - 2])
            s.append(g(r + (e1c if n == 2 else s[n - 3])))

        def s_(n):
            return s[n] if n >= 0 else conj(s[-n])

        c = list(zip(self.degrees, self._coefficients(cs)))
        sB = sB2 = None
        for k, ck in c:
            x = g(s[k] * (ck * k))
            sB = x if sB is None else g(sB + x)
            for l, cl in c:
                x = g(s_(k + l) * (ck * cl * (k * l)) + s_(k - l) * (ck * conj(cl) * (k * l)))
                sB2 = x if sB2 is None else g(sB2 + x)
        # (masked: outside P1 the coefficients are not bounded, there a = 1)
        trB = g(g.component.real(sB) * P1 * (-1.0))
        trB2 = g(g.component.real(sB2) * P1 * 0.5)
        a = g((6.0 * one + 4.0 * trB + g(trB * trB) - trB2) * (1.0 / 6.0))

        def disc(Q):
            # the discriminant of the traceless part A = Q - tr(Q)/3:
            # -4 p^3 - 27 q^2, p = -tr(A^2)/2, q = -tr(A^3)/3
            A = g(Q - identity * g(g.trace(Q) * (1.0 / 3.0)))
            A2 = g(A * A)
            p = g(g.trace(A2) * (-0.5))
            q = g(g.trace(g(A2 * A)) * (-1.0 / 3.0))
            return g(g(g(p * p) * p) * (-4.0) - g(q * q) * 27.0)

        # (shift-invariant for numerical stability: no cancellation of O(1)
        # terms when eigenvalues of P are close; check stability before
        # changing this form)

        log = g.component.log
        v = log(a) + log(g(-1.0 * disc(Pp))) - log(g(-1.0 * disc(P)))
        return g(g.component.real(g(v * P1)))

    def log_det_jacobian(self, fields):
        # the real field log det J(x) (zero outside P1); a node for nodes
        return self._log_det(fields)

    def action_log_det_jacobian(self):
        return spectral_loop_action_log_det_jacobian(self)


class spectral_loop_action_log_det_jacobian(differentiable_functional):
    # -sum_x log det J(x): one cached node graph per transport (leaves swapped
    # per call); its gradient w.r.t. the links and the coefficient fields
    def __init__(self, step):
        self.step = step

    def __call__(self, fields):
        return -g.sum(self.step.log_det_jacobian(fields)).real

    def gradient(self, fields, dfields):
        _, _, _, _, graphs = self.step.transport(fields)
        if "action" not in graphs:
            leaves = [g.ad.reverse.node(x) for x in fields]
            S = g.component.real(g.sum(self.step.log_det_jacobian(leaves))) * (-1.0)
            graphs["action"] = (S.functional(*leaves), leaves)
        functional, leaves = graphs["action"]
        gradients = functional.gradient(fields, dfields)
        # (the cached graph keeps no fields of its last pass)
        for x in leaves:
            x.value = None
            x.gradient = None
        # (a field the log det does not depend on gets a zero)
        return [g.group.zero(d) if x is None else x for x, d in zip(gradients, dfields)]
