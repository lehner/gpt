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
    return d.otype.inner_product(d, d) / d.otype.inner_product(d_last, d_last)


def polak_ribiere(d, d_last):
    ip_dd = d.otype.inner_product(d, d)
    ip_dl = d.otype.inner_product(d, d_last)
    ip_ll = d.otype.inner_product(d_last, d_last)
    return max([0.0, (ip_dd - ip_dl) / ip_ll])


class non_linear_cg(base_iterative):
    @g.params_convention(
        eps=1e-8,
        maxiter=1000,
        step=1e-3,
        log_functional_every=10,
        line_search=line_search_quadratic,
        beta=fletcher_reeves,
    )
    def __init__(self, params):
        super().__init__()
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.step = params["step"]
        self.nf = params["log_functional_every"]
        self.line_search = params["line_search"]
        self.beta = params["beta"]

    def __call__(self, f, df):
        @self.timed_function
        def opt(x, t):
            d_last = None
            s_last = None
            for i in range(self.maxiter):
                d = df(x)

                if i == 0:
                    beta = 0
                    s = d
                else:
                    beta = self.beta(d, d_last)
                    s = g(d + beta * s_last)

                c = self.line_search(s, x, d, df, -self.step)

                x @= g.group.compose(-self.step * c * s, x)

                rs = (g.norm2(d) / d.grid.gsites / d.otype.nfloats) ** 0.5

                self.log_convergence(i, rs, self.eps)

                if i % self.nf == 0:
                    v = f(x)
                    self.log(
                        f"iteration {i}: f(x) = {v:.15e}, |df|/sqrt(dof) = {rs:e}, beta = {beta}, step = {c*self.step}"
                    )

                if rs <= self.eps:
                    v = f(x)
                    self.log(
                        f"converged in {i+1} iterations: f(x) = {v:.15e}, |df|/sqrt(dof) = {rs:e}"
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
