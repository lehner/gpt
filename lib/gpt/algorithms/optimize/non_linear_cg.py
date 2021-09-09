#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.algorithms.optimize import line_search_quadratic


def fletcher_reeves(d, d_last):
    ip_dd = g.group.inner_product(d, d)
    ip_ll = g.group.inner_product(d_last, d_last)
    return ip_dd / ip_ll


def polak_ribiere(d, d_last):
    ip_dd = g.group.inner_product(d, d)
    ip_dl = g.group.inner_product(d, d_last)
    ip_ll = g.group.inner_product(d_last, d_last)
    return max([0.0, (ip_dd - ip_dl) / ip_ll])


class non_linear_cg(base_iterative):
    @g.params_convention(
        eps=1e-8,
        maxiter=1000,
        step=1e-3,
        log_functional_every=10,
        line_search=line_search_quadratic,
        beta=fletcher_reeves,
        max_abs_step=1.0,
    )
    def __init__(self, params):
        super().__init__()
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.step = params["step"]
        self.nf = params["log_functional_every"]
        self.line_search = params["line_search"]
        self.beta = params["beta"]
        self.max_abs_step = params["max_abs_step"]

    def __call__(self, f):
        @self.timed_function
        def opt(x, dx, t):
            x = g.util.to_list(x)
            dx = g.util.to_list(dx)
            d_last = None
            s_last = None
            for i in range(self.maxiter):
                d = f.gradient(x, dx)
                assert isinstance(d, list)

                if i == 0:
                    beta = 0
                    s = d
                else:
                    beta = self.beta(d, d_last)
                    for nu in range(len(s)):
                        s[nu] = g(d[nu] + beta * s_last[nu])

                next_step = (
                    self.line_search(s, x, dx, d, f.gradient, -self.step) * self.step
                )
                if abs(next_step) > self.max_abs_step:
                    self.log(f"max_abs_step adjustment for step = {next_step}")
                    next_step *= self.max_abs_step / abs(next_step)
                    beta = 0

                for nu, x_mu in enumerate(dx):
                    x_mu @= g.group.compose(-next_step * s[nu], x_mu)

                rs = (
                    sum(g.norm2(d)) / sum([s.grid.gsites * s.otype.nfloats for s in d])
                ) ** 0.5

                self.log_convergence(i, rs, self.eps)

                if i % self.nf == 0:
                    self.log(
                        f"iteration {i}: f(x) = {f(x):.15e}, |df|/sqrt(dof) = {rs:e}, beta = {beta}, step = {next_step}"
                    )

                if rs <= self.eps:
                    self.log(
                        f"converged in {i+1} iterations: f(x) = {f(x):.15e}, |df|/sqrt(dof) = {rs:e}"
                    )
                    return True

                # keep last search direction
                d_last = d
                s_last = s

            self.log(
                f"NOT converged in {i+1} iterations;  |df|/sqrt(dof) = {rs:e} / {self.eps:e}"
            )
            return False

        return opt
