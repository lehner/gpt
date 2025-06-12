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
from gpt.core.group import differentiable_functional


class locally_coherent_functional(differentiable_functional):
    def __init__(self, inner, block):
        self.inner = inner
        self.block = block

    def reduce(self, fields):
        n = len(fields)
        assert n % 2 == 0
        n //= 2

        right = self.block.embed(fields[n : 2 * n])

        return [g(g.group.compose(x, y)) for x, y in zip(fields[0:n], right)]

    def __call__(self, fields):
        return self.inner(self.reduce(fields))

    def gradient(self, fields, dfields):
        n = len(fields)
        assert n % 2 == 0
        n //= 2
        left = fields[0:n]
        right = fields[n : 2 * n]

        # f(left right)
        # left derivative is like original: f(idA left right)
        # right derivative is: f(left idA right  ) = f(idA2 left right)
        # with dA2 = left dA left^dag

        indices = [
            mu for mu in range(n) if dfields[mu] in fields or dfields[mu + n] in fields
        ]

        r = self.reduce(fields)

        inner_gradient = self.inner.gradient(r, [r[mu] for mu in indices])

        dSdA = []

        for f in dfields:
            if f in left:
                mu = left.index(f)
                igi = indices.index(mu)
                dSdA.append(inner_gradient[igi])
            elif f in right:
                mu = right.index(f)
                igi = indices.index(mu)
                fgrad = g(g.group.inverse(left[mu]) * inner_gradient[igi] * left[mu])
                fgrad.otype = left[0].otype
                gr = self.block.sum(fgrad)
                gr.otype = inner_gradient[igi].otype
                dSdA.append(gr)
            else:
                assert False

        return dSdA
