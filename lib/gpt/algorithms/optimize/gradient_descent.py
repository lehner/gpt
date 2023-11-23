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
from gpt.algorithms.optimize import line_search_none


class gradient_descent(base_iterative):
    @g.params_convention(
        eps=1e-8,
        maxiter=1000,
        step=1e-3,
        log_functional_every=10,
        line_search=line_search_none,
    )
    def __init__(self, params):
        super().__init__()
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.step = params["step"]
        self.nf = params["log_functional_every"]
        self.line_search = params["line_search"]

    def __call__(self, f):
        @self.timed_function
        def opt(x, dx, t):
            x = g.util.to_list(x)
            dx = g.util.to_list(dx)
            for i in range(self.maxiter):
                d = f.gradient(x, dx)

                c = self.line_search(d, x, dx, d, f.gradient, -self.step)

                for nu, x_mu in enumerate(dx):
                    x_mu @= g.group.compose(-self.step * c * d[nu], x_mu)

                if self.eps is None:
                    continue

                rs = (sum(g.norm2(d)) / sum([s.nfloats() for s in d])) ** 0.5

                self.log_convergence(i, rs, self.eps)

                if i % self.nf == 0:
                    self.log(
                        f"iteration {i}: f(x) = {f(x):.15e}, |df|/sqrt(dof) = {rs:e}, step = {c*self.step}"
                    )

                if rs <= self.eps:
                    self.log(
                        f"converged in {i+1} iterations: f(x) = {f(x):.15e}, |df|/sqrt(dof) = {rs:e}"
                    )
                    return True

            if self.eps is not None:
                self.log(
                    f"NOT converged in {i+1} iterations;  |df|/sqrt(dof) = {rs:e} / {self.eps:e}"
                )
            return False

        return opt
