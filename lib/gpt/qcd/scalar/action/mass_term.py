#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Mattia Bruno
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
from gpt.core.group import differentiable_functional


class mass_term(differentiable_functional):
    def __init__(self, m=1.0):
        self.m = m
        self.__name__ = f"mass_term({m})"

    def __call__(self, pi):
        return g.group.inner_product(pi, pi) * self.m * 0.5

    def draw(self, pi, rng):
        rng.normal_element(pi, scale=self.m**-0.5)
        return self.__call__(pi)

    @differentiable_functional.multi_field_gradient
    def gradient(self, pi, dpi):
        dS = []
        for _pi in dpi:
            i = pi.index(_pi)
            dS.append(g(self.m * pi[i]))
        return dS


class fourier_mass_term(differentiable_functional):
    def __init__(self, fourier_sqrt_mass_field):
        self.n = len(fourier_sqrt_mass_field)

        self.fourier_sqrt_mass_field = fourier_sqrt_mass_field
        self.fourier_mass_field = [
            [g.lattice(fourier_sqrt_mass_field[0][0]) for i in range(self.n)] for j in range(self.n)
        ]
        for i in range(self.n):
            for j in range(self.n):
                r = self.fourier_mass_field[i][j]
                r[:] = 0
                for l in range(self.n):
                    r += self.fourier_sqrt_mass_field[i][l] * self.fourier_sqrt_mass_field[l][j]

        # generate inverse
        self.fourier_inv_sqrt_mass_field = [
            [g.lattice(fourier_sqrt_mass_field[0][0]) for i in range(self.n)] for j in range(self.n)
        ]
        sqrt = np.moveaxis(
            np.array(
                [
                    [self.fourier_sqrt_mass_field[i][j][:][:, 0] for i in range(self.n)]
                    for j in range(self.n)
                ]
            ),
            2,
            0,
        )
        isqrt = np.linalg.inv(sqrt)
        for i in range(self.n):
            for j in range(self.n):
                self.fourier_inv_sqrt_mass_field[i][j][:] = np.ascontiguousarray(isqrt[:, j, i])

        self.__name__ = f"fourier_mass_term({self.n} x {self.n})"
        L = self.fourier_mass_field[0][0].grid.gdimensions
        self.scale_unitary = float(np.prod(L)) ** 0.5
        self.fft = g.fft()

    def __call__(self, pi):
        A = 0.0
        fft_pi = g(self.fft * pi)
        for mu in range(self.n):
            for nu in range(self.n):
                x = g(self.scale_unitary**2 * self.fourier_mass_field[mu][nu] * fft_pi[nu])
                A += g.inner_product(fft_pi[mu], x)
        return A.real

    def draw(self, pi, rng):
        # P(pi) = e^{-sum(trace(pi^dag scale^2 fft^dag mass fft pi)) * 2 / 2}    # fft^dag fft = 1/scale^2 ; inv(fft) = fft^dag / scale^2
        rng.normal_element(pi)

        value = g.group.inner_product(pi, pi) * 0.5

        pi_mom = [g(self.scale_unitary * self.fft * p) for p in pi]
        for mu in range(self.n):
            r = g.lattice(pi_mom[mu])
            r[:] = 0
            for nu in range(self.n):
                r += self.fourier_inv_sqrt_mass_field[mu][nu] * pi_mom[nu]
            pi[mu] @= g.inv(self.fft) * r / self.scale_unitary

        return value

    @differentiable_functional.multi_field_gradient
    def gradient(self, pi, dpi):
        dS = []
        fft_pi = g(self.fft * pi)
        for _pi in dpi:
            mu = pi.index(_pi)
            ret = g.lattice(pi[mu])
            ret[:] = 0

            for nu in range(self.n):
                ret += self.fourier_mass_field[mu][nu] * fft_pi[nu]

            ret @= g.inv(self.fft) * ret

            ret = g(0.5 * (ret + g.adj(ret)))
            dS.append(ret)

        return dS


class general_mass_term(differentiable_functional):
    def __init__(self, M, sqrt_M, M_projected_gradient):
        self.M = M
        self.sqrt_M = sqrt_M
        self.inv_sqrt_M = sqrt_M.inv()
        self.M_projected_gradient = M_projected_gradient
        # Need:
        # - sqrt_M^dag = sqrt_M
        # - sqrt_M^2 = M
        # - M_projected_gradient = D[vec^dag M(e^{iTa eps} U) vec, eps] iTa

    def __call__(self, fields):
        # fields = U + pi
        n = len(fields)
        assert n % 2 == 0
        n //= 2
        pi = fields[n:]
        pi_prime = self.M(fields)[n:]
        A = 0.0
        for mu in range(n):
            A += g.inner_product(pi[mu], pi_prime[mu])
        return A.real

    def draw(self, fields, rng):
        # fields = U + pi
        n = len(fields)
        assert n % 2 == 0
        n //= 2
        pi = fields[n:]

        # P(pi) = e^{-pi^dag sqrt_M^dag sqrt_M pi * 2 / 2}
        rng.normal_element(pi)  # pi = sqrt(2) sqrt_M pi_desired -> pi_desired = inv_sqrt_M pi

        value = g.group.inner_product(pi, pi) * 0.5

        fields_prime = self.inv_sqrt_M(fields)
        pi_prime = fields_prime[n:]

        for mu in range(n):
            pi_prime[mu].otype = pi[mu].otype
            pi[mu] @= g.project(pi_prime[mu], "defect")

        return value

    @differentiable_functional.multi_field_gradient
    def gradient(self, fields, dfields):
        # fields = U + pi
        n = len(fields)
        assert n % 2 == 0
        n //= 2
        U = fields[0:n]
        pi = fields[n:]

        dS = []

        dU = [f for f in dfields if f in U]
        if len(dU) > 0:
            grad_U = self.M_projected_gradient(U, pi)
        else:
            grad_U = None

        dpi = [f for f in dfields if f in pi]
        if len(dpi) > 0:
            pi_prime = self.M(fields)[n:]
        else:
            pi_prime = None

        for _field in dfields:
            mu = fields.index(_field)

            if mu >= n:
                mu -= n
                # pi[mu]
                ret = pi_prime[mu]
                ret = g(g.project(ret, "defect"))
            else:
                # U[mu]
                ret = grad_U[mu]

            dS.append(ret)

        return dS
