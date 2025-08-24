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
class expression:
    discard = 2

    def __init__(self, graph):
        self.graph = graph

    def __rmul__(self, other):
        if isinstance(other, (float, complex, int)):
            return expression([(a[0] * other, a[1], a[2]) for a in self.graph])
        return other * self

    def __mul__(self, other):
        if isinstance(other, (float, complex, int)):
            return expression([(a[0] * other, a[1], a[2]) for a in self.graph])
        other = other.unique_indices(self.indices())
        return expression(
            [
                (a[0] * b[0], a[1] + b[1], a[2] + b[2])
                for a in self.graph
                for b in other.graph
                if a[1] + b[1] <= expression.discard
            ]
        )

    def __matmul__(self, other):
        return expression(
            [
                (a[0] * b[0], a[1] + b[1], a[2] + b[2])
                for a in self.graph
                for b in other.graph
                if a[1] + b[1] <= expression.discard
            ]
        )

    def indices(self):
        indices = set()
        for c, p, fac in self.graph:
            for f in fac:
                if f[1][0] == "*":
                    indices.add(int(f[1][1:]))
        return indices

    def replace_coordinate(self, src, dst):
        return expression(
            [(a[0], a[1], [(x[0], dst if x[1] == src else x[1]) for x in a[2]]) for a in self.graph]
        )

    def unique_indices(self, avoid):
        available = 0
        mine = self.indices()
        new = self
        for m in mine:
            if m in avoid:
                while available in mine or available in avoid:
                    available += 1
                new = new.replace_coordinate(f"*{m}", f"*{available}")
        return new

    def __add__(self, other):
        return expression(self.graph + other.graph)

    def __sub__(self, other):
        return self + (-1) * other

    def __pow__(self, n):
        assert isinstance(n, int)
        if n == 1:
            return self
        assert n > 1
        return self * self ** (n - 1)

    def __str__(self):
        s = ""
        for c, p, fac in self.graph:
            fs = f"({c})*e**{p}"
            for f in fac:
                fs = f"{fs} * {f[0]}({f[1]})"
            s = f"{s}+ {fs}\n"
        return s
