#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2024  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


class coupling(differentiable_functional):
    def __init__(self, omega, cartesian):
        self.omega = omega
        self.cartesian = cartesian
        self.__name__ = f"coupling({omega}, {cartesian})"

    def __call__(self, pi):
        n = len(pi)
        assert n % 2 == 0
        n //= 2
        a = pi[0:n]
        b = pi[n:]

        if self.cartesian:
            r = 0.0
            for i in range(n):
                diff = g(a[i] - b[i])
                r += g.inner_product(diff, diff) * self.omega * 0.5
            return r.real
        else:
            # a = e^{i dA} U0
            # b = e^{i dB} U0
            # tr(a b^dag) = tr(e^{i dA} e^{-i dB})
            #             = tr(1) + tr(dA dB) - 1/2 tr(dA^2) - 1/2 tr(dB^2) + O(d^3)
            #             = const - 1/2 * tr( (dA - dB)^2 ) + O(d^3)
            r = 0.0
            for i in range(n):
                r += g.sum(g.trace(g.identity(a[i]) - a[i] * g.adj(b[i]))) * self.omega
            return r.real

    @differentiable_functional.multi_field_gradient
    def gradient(self, pi, dpi):
        n = len(pi)
        assert n % 2 == 0
        n //= 2
        a = pi[0:n]
        b = pi[n:]

        dS = []
        for _pi in dpi:
            i = pi.index(_pi)
            if i < n:
                # da[i]
                if not self.cartesian:
                    dS_dpi = g(
                        self.omega
                        / 2.0
                        / 1j
                        * g.qcd.gauge.project.traceless_anti_hermitian(g(a[i] * g.adj(b[i])))
                    )
                    dS_dpi.otype = _pi.otype.cartesian()
                    dS.append(dS_dpi)
                else:
                    dS.append(g(self.omega * (a[i] - b[i]) * 0.5))
            else:
                i -= n
                # db[i]
                if not self.cartesian:
                    dS_dpi = g(
                        -self.omega
                        / 2.0
                        / 1j
                        * g.qcd.gauge.project.traceless_anti_hermitian(g(a[i] * g.adj(b[i])))
                    )
                    dS_dpi.otype = _pi.otype.cartesian()
                    dS.append(dS_dpi)
                else:
                    dS.append(g(self.omega * (b[i] - a[i]) * 0.5))
        return dS
