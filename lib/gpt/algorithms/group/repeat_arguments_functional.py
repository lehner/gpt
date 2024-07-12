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


class repeat_arguments_functional(differentiable_functional):
    def __init__(self, inner, indices):
        self.inner = inner
        self.indices = indices
        self.n = max(indices) + 1

    def reduce(self, fields):
        r = [None] * len(self.indices)
        assert len(fields) == self.n
        for i in range(len(self.indices)):
            r[i] = g.copy(fields[self.indices[i]])
        return r

    def __call__(self, fields):
        r = self.reduce(fields)
        return self.inner(r)

    def gradient(self, fields, dfields):
        r = self.reduce(fields)

        # TODO: figure out which gradients are needed and only calculate those
        grad = self.inner.gradient(r, r)

        reduced_grad = [None] * self.n

        for i in range(len(self.indices)):
            j = self.indices[i]
            if reduced_grad[j] is None:
                reduced_grad[j] = grad[i]
            else:
                reduced_grad[j] += grad[i]

        return [reduced_grad[fields.index(d)] for d in dfields]
