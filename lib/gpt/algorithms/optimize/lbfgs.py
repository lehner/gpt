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
from gpt.algorithms.optimize.optimizer import optimizer
from gpt.algorithms.optimize.adam import set_element, nfloats


def _axpy(a, c, b):
    # a + c b for numbers, numpy arrays and lattices
    if g.util.is_num(a) or isinstance(a, np.ndarray):
        return a + c * b
    return g(a + c * b)


def _ip(a, b):
    return float(np.real(g.group.inner_product(a, b)))


def _cubic_minimum(a, fa, da, b, fb, db):
    # the minimizer of the cubic through (a, fa, da) and (b, fb, db), or None
    d1 = da + db - 3 * (fa - fb) / (a - b)
    d2 = d1**2 - da * db
    if d2 < 0:
        return None
    d2 = np.sign(b - a) * d2**0.5
    x = b - (b - a) * (db + d2 - d1) / (db - da + 2 * d2)
    return x if np.isfinite(x) else None


class lbfgs(optimizer):
    """Limited-memory BFGS in GPT operations (no host copies of lattices):
    the direction from the last `memory` pairs of steps s and gradient
    changes y (two-loop recursion with g.group.inner_product), a line search
    for the strong Wolfe conditions (c1, c2) along it, and updates
    x <- g.group.compose(alpha p, x) of every field type (additive fields:
    exact L-BFGS; group fields: steps in the algebra, as non_linear_cg).
    The first step of a run (no pairs yet) has length `step` (in the norm of
    the gradient direction); afterwards the unit step is tried first.
    Converged when |df| / sqrt(dof) <= eps.  The pairs are kept across the
    calls of a run (opt.on).  failure_value: if not None, a RuntimeError while
    evaluating f or its gradient at a trial point (e.g. a map that cannot be
    inverted there) counts as a failed trial (the line search steps back);
    otherwise it propagates."""

    @g.params_convention(
        eps=1e-8,
        maxiter=1000,
        memory=10,
        step=1e-3,
        c1=1e-4,
        c2=0.9,
        max_line_search=20,
        log_functional_every=10,
        failure_value=None,
    )
    def __init__(self, params):
        super().__init__()
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.memory = params["memory"]
        self.step = params["step"]
        self.c1 = params["c1"]
        self.c2 = params["c2"]
        self.max_line_search = params["max_line_search"]
        self.nf = params["log_functional_every"]
        self.failure_value = params["failure_value"]

    def new_state(self):
        return {"pairs": []}

    def direction(self, d, pairs):
        # -H d with the L-BFGS inverse Hessian H of the pairs (s, y, 1 / <s, y>)
        q = list(d)
        alphas = []
        for s, y, rho in reversed(pairs):
            a = rho * _ip(s, q)
            q = [_axpy(qi, -a, yi) for qi, yi in zip(q, y)]
            alphas.append(a)
        s, y, rho = pairs[-1]
        gamma = 1.0 / (rho * _ip(y, y))
        r = [gamma * qi if g.util.is_num(qi) else g(gamma * qi) for qi in q]
        for (s, y, rho), a in zip(pairs, reversed(alphas)):
            b = rho * _ip(y, r)
            r = [_axpy(ri, a - b, si) for ri, si in zip(r, s)]
        return [-ri if g.util.is_num(ri) or isinstance(ri, np.ndarray) else g(-ri) for ri in r]

    def iterate(self, f, x, dx_indices, state, t):
        pairs = state["pairs"]

        def fields():
            return [x[i] for i in dx_indices]

        def evaluate():
            try:
                return complex(f(x)).real, f.gradient(x, fields())
            except RuntimeError as e:
                if self.failure_value is None:
                    raise
                self.log(f"trial point rejected ({e})")
                return None

        r = evaluate()
        if r is None:
            raise RuntimeError("lbfgs: the starting point cannot be evaluated")
        value, d = r
        dof = sum(nfloats(v) for v in d)
        for i in range(self.maxiter):
            rs = (_ip(d, d) / dof) ** 0.5
            self.log_convergence(i, rs, self.eps)
            if i % self.nf == 0:
                self.log(f"iteration {i}: f(x) = {value:.15e}, |df|/sqrt(dof) = {rs:e}")
            if rs <= self.eps:
                self.log(
                    f"converged in {i} iterations: f(x) = {value:.15e}, |df|/sqrt(dof) = {rs:e}"
                )
                return True

            if pairs:
                p = self.direction(d, pairs)
                alpha = 1.0
            else:
                p = [-di if g.util.is_num(di) or isinstance(di, np.ndarray) else g(-di) for di in d]
                alpha = self.step / (_ip(d, d) ** 0.5)
            dphi0 = _ip(d, p)
            if dphi0 >= 0:
                # not a descent direction: restart from the gradient
                self.log("not a descent direction: pairs dropped")
                pairs.clear()
                p = [-di if g.util.is_num(di) or isinstance(di, np.ndarray) else g(-di) for di in d]
                alpha = self.step / (_ip(d, d) ** 0.5)
                dphi0 = _ip(d, p)

            x0 = [
                v if g.util.is_num(v) or isinstance(v, np.ndarray) else g.copy(v) for v in fields()
            ]

            def phi(a):
                # f and its derivative along p at x0 + a p (x set to it)
                for k, j in enumerate(dx_indices):
                    set_element(x, j, g.group.compose(_scaled(a, p[k]), x0[k]))
                r = evaluate()
                if r is None:
                    return None
                return r[0], _ip(r[1], p), r[1]

            result = self.line_search(phi, value, dphi0, alpha)
            if result is None:
                for k, j in enumerate(dx_indices):
                    set_element(x, j, x0[k])
                self.log(f"line search failed in iteration {i}: f(x) = {value:.15e}")
                return False
            alpha, value, d_new = result

            s = [_scaled(alpha, pk) for pk in p]
            y = [_axpy(a, -1.0, b) for a, b in zip(d_new, d)]
            sy = _ip(s, y)
            if sy > 1e-12 * (_ip(s, s) * _ip(y, y)) ** 0.5:
                pairs.append((s, y, 1.0 / sy))
                if len(pairs) > self.memory:
                    pairs.pop(0)
            d = d_new

        rs = (_ip(d, d) / dof) ** 0.5
        if self.maxiter > 1:
            self.log(
                f"NOT converged in {self.maxiter} iterations;  |df|/sqrt(dof) = {rs:e} / {self.eps:e}"
            )
        return False

    def line_search(self, phi, phi0, dphi0, alpha):
        # the strong Wolfe conditions (Nocedal and Wright, algorithms 3.5 and
        # 3.6, with cubic interpolation); a failed trial point shrinks the
        # step.  Returns (alpha, f, gradient) at the accepted point, which is
        # the point x is left at, or None.
        c1, c2 = self.c1, self.c2

        def zoom(lo, hi, n):
            # lo, hi: (alpha, f, df); lo satisfies the sufficient decrease
            for _ in range(n):
                a = None
                if np.isfinite(hi[1]):
                    a = _cubic_minimum(*lo, *hi)
                width = hi[0] - lo[0]
                if a is None or not (
                    min(lo[0], hi[0]) + 0.1 * abs(width)
                    <= a
                    <= max(lo[0], hi[0]) - 0.1 * abs(width)
                ):
                    a = lo[0] + 0.5 * width
                r = phi(a)
                if r is None:
                    hi = (a, np.inf, np.inf)
                    continue
                fa, da, grad = r
                if fa > phi0 + c1 * a * dphi0 or fa >= lo[1]:
                    hi = (a, fa, da)
                else:
                    if abs(da) <= -c2 * dphi0:
                        return a, fa, grad
                    if da * (hi[0] - lo[0]) >= 0:
                        hi = lo
                    lo = (a, fa, da)
            return None

        prev = (0.0, phi0, dphi0)
        for n in range(self.max_line_search):
            r = phi(alpha)
            if r is None:
                alpha = prev[0] + 0.5 * (alpha - prev[0])
                continue
            fa, da, grad = r
            if fa > phi0 + c1 * alpha * dphi0 or (n > 0 and fa >= prev[1]):
                result = zoom(prev, (alpha, fa, da), self.max_line_search - n)
                break
            if abs(da) <= -c2 * dphi0:
                return alpha, fa, grad
            if da >= 0:
                result = zoom((alpha, fa, da), prev, self.max_line_search - n)
                break
            prev = (alpha, fa, da)
            alpha *= 2.0
        else:
            return None
        return result


def _scaled(a, x):
    if g.util.is_num(x) or isinstance(x, np.ndarray):
        return a * x
    return g(a * x)
