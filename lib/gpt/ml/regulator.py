#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.core.group import differentiable_functional


class L2(differentiable_functional):
    def __init__(self, lam, indices):
        self.lam = lam
        self.indices = indices

    def __call__(self, a):
        a = g.util.to_list(a)
        return (self.lam / 2) * sum([g.norm2(a[i]) for i in self.indices])

    def gradient(self, a, da):
        return [g((self.lam if a.index(x) in self.indices else 0.0) * x) for x in da]


class L1(differentiable_functional):
    def __init__(self, lam, indices):
        self.lam = lam
        self.indices = indices

    def __call__(self, a):
        a = g.util.to_list(a)
        r = 0.0
        for i in self.indices:
            x = g(
                g.component.sqrt(g.component.abs(g.component.real(a[i])))
                + 1j * g.component.sqrt(g.component.abs(g.component.imag(a[i])))
            )
            r += g.sum(
                g.trace(g.adj(x) * x)
            )
        return r * self.lam

    def gradient(self, a, da):
        dabs = g.component.drelu(-1)
        return [
            g(
                (self.lam if a.index(x) in self.indices else 0.0)
                * (dabs(g.component.real(x)) + 1j * dabs(g.component.imag(x)))
            ) for x in da
        ]
