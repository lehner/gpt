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
        rng.normal_element(pi, scale=self.m**0.5)
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
        fft_pi = g(self.fft * pi)
        for mu in range(self.n):
            pi[mu][:] = 0
            for nu in range(self.n):
                pi[mu] += self.fourier_sqrt_mass_field[mu][nu] * fft_pi[nu]
            pi[mu] @= g.inv(self.fft) * pi[mu]
            pi[mu] @= g(0.5 * (pi[mu] + g.adj(pi[mu])))
        return self.__call__(pi)

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
