#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
#    Reference: https://arxiv.org/pdf/1412.6980.pdf
#
import gpt as g
from gpt.algorithms import base_iterative


class adam(base_iterative):
    @g.params_convention(
        eps=1e-8,
        maxiter=1000,
        alpha=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps_regulator=1e-8,
        log_functional_every=10,
    )
    def __init__(self, params):
        super().__init__()
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.alpha = params["alpha"]
        self.beta1 = params["beta1"]
        self.beta2 = params["beta2"]
        self.eps_regulator = params["eps_regulator"]
        self.nf = params["log_functional_every"]

    def __call__(self, f):
        class context:
            m = None
            t = 0

        @self.timed_function
        def opt(x, dx, t):
            x = g.util.to_list(x)
            dx = g.util.to_list(dx)

            # momentum vectors
            if context.m is None:
                context.m = [a.new() for a in dx]
                context.v = [a.new() for a in dx]
                context.mhat = [a.new() for a in dx]
                context.vhat = [a.new() for a in dx]
                context.epsfield = [a.new() for a in dx]

                for a in context.m + context.v:
                    a[:] = 0

                for a in context.epsfield:
                    a[:] = self.eps_regulator

            for i in range(self.maxiter):
                context.t += 1

                # x = theta(t-1)
                gt = f.gradient(x, dx)
                gt2 = []
                for a in gt:
                    ar = g(g.component.real(a))
                    ai = g(g.component.imag(a))
                    gt2.append(g(g.component.multiply(ar, ar) + 1j * g.component.multiply(ai, ai)))

                for nu in range(len(dx)):
                    context.m[nu] @= self.beta1 * context.m[nu] + (1 - self.beta1) * gt[nu]
                    context.v[nu] @= self.beta2 * context.v[nu] + (1 - self.beta2) * gt2[nu]

                    context.mhat[nu] @= (1.0 / (1.0 - self.beta1**context.t)) * context.m[nu]
                    context.vhat[nu] @= (1.0 / (1.0 - self.beta2**context.t)) * context.v[nu]

                    vhat_nu_real = g(g.component.sqrt(g.component.real(context.vhat[nu])))
                    vhat_nu_imag = g(g.component.sqrt(g.component.imag(context.vhat[nu])))

                    reg_mhat_real = g.component.multiply(
                        g.component.real(context.mhat[nu]),
                        g.component.inv(context.epsfield[nu] + vhat_nu_real),
                    )
                    reg_mhat_imag = g.component.multiply(
                        g.component.imag(context.mhat[nu]),
                        g.component.inv(context.epsfield[nu] + vhat_nu_imag),
                    )

                    # make sure object type is correct
                    tmp = gt[nu].new()
                    tmp @= -self.alpha * (reg_mhat_real + 1j * reg_mhat_imag)

                    dx[nu] @= g.group.compose(tmp, dx[nu])

                rs = (sum(g.norm2(gt)) / sum([s.nfloats() for s in gt])) ** 0.5

                self.log_convergence(i, rs, self.eps)

                if i % self.nf == 0:
                    self.log(
                        f"iteration {i}: f(x) = {f(x):.15e}, |df|/sqrt(dof) = {rs:e}, alpha = {self.alpha}, beta1 = {self.beta1}, beta2 = {self.beta2}"
                    )

                if rs <= self.eps:
                    self.log(
                        f"converged in {i+1} iterations: f(x) = {f(x):.15e}, |df|/sqrt(dof) = {rs:e}"
                    )
                    return True

            self.log(f"NOT converged in {i+1} iterations;  |df|/sqrt(dof) = {rs:e} / {self.eps:e}")
            return False

        return opt
