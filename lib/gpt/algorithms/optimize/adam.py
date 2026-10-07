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
import numpy as np
from gpt.algorithms.optimize.optimizer import optimizer


def new(a):
    if g.util.is_num(a):
        return a
    elif isinstance(a, np.ndarray):
        return np.copy(a)
    return a.new()


def nfloats(a):
    if g.util.is_num(a):
        # treat as complex
        return 2
    elif isinstance(a, np.ndarray):
        return a.size * 2
    return a.nfloats()


def set_element(a, i, b):
    # fields and arrays are updated in place (others may hold them, e.g. the
    # functions of g.ml and the composites that share their parameters);
    # numbers are replaced
    if isinstance(a[i], (g.lattice, g.tensor)):
        if g.util.is_num(b):
            a[i][:] = b
        else:
            a[i] @= b
    elif isinstance(a[i], np.ndarray):
        a[i][...] = b
    else:
        a[i] = b


def set_value(a, b):
    for i in range(len(a)):
        set_element(a, i, b)


class adam(optimizer):
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

    def new_state(self):
        # the moments and the step count (kept across the calls of a run)
        class context:
            m = None
            t = 0

        return context

    def iterate(self, f, x, dx_indices, context, t):

        for i in range(self.maxiter):
            context.t += 1

            # x = theta(t-1)
            dx = [x[i] for i in dx_indices]
            gt = f.gradient(x, dx)
            gt2 = []

            # momentum vectors
            if context.m is None:
                context.m = [new(a) for a in gt]
                context.v = [new(a) for a in gt]
                context.mhat = [new(a) for a in gt]
                context.vhat = [new(a) for a in gt]
                context.epsfield = [new(a) for a in gt]

                set_value(context.m, 0)
                set_value(context.v, 0)
                set_value(context.epsfield, self.eps_regulator)
            else:
                # (numbers may change type, e.g. complex -> numpy.complex128)
                m0, g0 = context.m[0], gt[0]
                assert (g.util.is_num(m0) and g.util.is_num(g0)) or type(m0) is type(g0)

            for a in gt:
                ar = g(g.component.real(a))
                ai = g(g.component.imag(a))
                gt2.append(g(g.component.multiply(ar, ar) + 1j * g.component.multiply(ai, ai)))

            for nu in range(len(dx)):
                set_element(context.m, nu, self.beta1 * context.m[nu] + (1 - self.beta1) * gt[nu])
                set_element(context.v, nu, self.beta2 * context.v[nu] + (1 - self.beta2) * gt2[nu])

                set_element(context.mhat, nu, (1.0 / (1.0 - self.beta1**context.t)) * context.m[nu])
                set_element(context.vhat, nu, (1.0 / (1.0 - self.beta2**context.t)) * context.v[nu])

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
                tmp = [new(gt[nu])]
                set_element(tmp, 0, -self.alpha * (reg_mhat_real + 1j * reg_mhat_imag))
                set_element(x, dx_indices[nu], g.group.compose(tmp[0], x[dx_indices[nu]]))

            rs = (sum([g.norm2(x) for x in gt]) / sum([nfloats(s) for s in gt])) ** 0.5

            self.log_convergence(i, rs, self.eps)

            if i % self.nf == 0:
                self.log(
                    f"iteration {i}: f(x) = {f(x):.15e}, |df|/sqrt(dof) = {rs:e}, alpha = {self.alpha}, beta1 = {self.beta1}, beta2 = {self.beta2}"
                )

            if rs <= self.eps:
                self.log(
                    f"converged in {i + 1} iterations: f(x) = {f(x):.15e}, |df|/sqrt(dof) = {rs:e}"
                )
                return True

        if self.maxiter > 1:
            self.log(
                f"NOT converged in {i + 1} iterations;  |df|/sqrt(dof) = {rs:e} / {self.eps:e}"
            )
        return False
