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
from gpt.algorithms import base_iterative


class fixed_point(base_iterative):
    """The iteration control of a fixed-point problem x = Phi(x):

      fp = g.algorithms.nonlinear.fixed_point(eps=1e-14, maxiter=100)
      fp(x, step)                    # x: a list of fields, updated in place
      fp(x, step, accelerated_step)  # switch to accelerated steps if slow

    step(x) replaces x by Phi(x) (in place).  Each iteration records the
    change^2 per site, sum_i |x_i - x_i,last|^2 / gsites_i (history), and
    the iteration has converged once it is below eps^2.  If the iteration
    contracts slowly (the rate of the change over the last four steps
    above accelerate_rate; None: never), it continues with
    accelerated_step(x) (e.g. Newton steps, which also converge where Phi is
    not a contraction).  Not converged after maxiter iterations: a
    RuntimeError reporting the contraction rate (close to 1: Phi is close to
    not being a contraction; a stall below 1: rounding), or with
    raise_on_failure=False the last iterate (converged is False).
    monitor(i, change) is called after every iteration (change: the change
    per site, not squared).  After a call: history, converged and
    accelerated_iterations (the number of accelerated steps).
    fixed_point.newton(residual, solve) makes Newton steps as
    accelerated_step."""

    @g.params_convention(eps=1e-14, maxiter=100, accelerate_rate=0.5, raise_on_failure=True)
    def __init__(self, params):
        super().__init__()
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.accelerate_rate = params["accelerate_rate"]
        self.raise_on_failure = params["raise_on_failure"]

    def __call__(self, x, step, accelerated_step=None, monitor=None, name=None):
        x = g.util.to_list(x)
        name = self.name if name is None else name
        self.history = []
        self.accelerated_iterations = 0
        self.converged = False
        accelerated = False
        for i in range(self.maxiter):
            x_last = g.copy(x)
            if accelerated:
                accelerated_step(x)
                self.accelerated_iterations += 1
            else:
                step(x)
            change2 = sum(g.norm2(a - b) / a.grid.gsites for a, b in zip(x, x_last))
            self.log_convergence(i, change2, self.eps**2)
            if monitor is not None:
                monitor(i, change2**0.5)
            if change2 < self.eps**2:
                self.converged = True
                self.log(
                    f"converged in {i + 1} iterations ({self.accelerated_iterations} accelerated)"
                )
                return x
            if (
                not accelerated
                and accelerated_step is not None
                and self.accelerate_rate is not None
            ):
                if len(self.history) >= 5:
                    accelerated = self.rate(4) > self.accelerate_rate
        if self.raise_on_failure:
            raise RuntimeError(
                f"{name} did not converge: change^2 per site {self.history[-1]:.3e} > "
                f"{self.eps**2:.3e} after {self.maxiter} iterations (contraction rate {self.rate(10)})"
            )
        self.log(
            f"NOT converged in {self.maxiter} iterations: change^2 per site {self.history[-1]:.3e}"
        )
        return x

    @staticmethod
    def newton(residual, solve, compose=None, max_halvings=8):
        """A Newton step for residual(x) = 0 as accelerated_step (x and the
        residual: lists of fields): x <- compose(d, x) with the direction
        d = solve(x, r) for r = residual(x) (e.g. d = J^-1 r for the Jacobian
        J of the residual along compose), halved while |residual| grows (at
        most max_halvings times).  compose(d_i, x_i) defaults to
        g.group.compose."""
        compose = g.group.compose if compose is None else compose

        def step(x):
            r = residual(x)
            d = solve(x, r)
            r2 = sum(g.norm2(ri) for ri in r)
            x0 = g.copy(x)
            for _ in range(max_halvings):
                x_new = [g(compose(di, xi)) for di, xi in zip(d, x0)]
                if sum(g.norm2(ri) for ri in residual(x_new)) < r2:
                    break
                d = [g(0.5 * di) for di in d]
            for xi, yi in zip(x, x_new):
                xi @= yi

        return step

    def rate(self, n):
        # the contraction rate per iteration (of the change, not squared)
        # over the last n iterations, or None if there are fewer
        if len(self.history) <= n:
            return None
        return (self.history[-1] / self.history[-1 - n]) ** (0.5 / n)
