#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import numpy as np


def redundant_operators(ops):
    basis = set(ops[0].basis())
    for o in ops[1:]:
        basis = basis.union(o.basis())
    basis = list(basis)
    v = [np.array(o.vector(basis), dtype=np.complex128) for o in ops]
    redundant = [False] * len(ops)
    for i in range(len(ops)):
        vv = v[i]
        for j in range(i):
            vv -= v[j] * np.dot(np.conjugate(v[j]), v[i]) / np.dot(np.conjugate(v[j]), v[j])
            eps = np.linalg.norm(vv)
            if eps < 1e-14:
                redundant[i] = True
                break
    return redundant


def reduce_operators(ops):
    r = redundant_operators(ops)
    return [ops[i] for i in range(len(ops)) if not r[i]]


class operator:
    def __init__(self, data):
        if isinstance(data, tuple):
            data = [(1, data)]
        self.data = data

    def __str__(self):
        r = ""
        for w, m in self.data:
            if r != "":
                r = r + f" + ({w}) * O({m})\n"
            else:
                r = f"   ({w}) * O({m})\n"
        return r

    def __rmul__(self, other):
        return operator([(w * other, m) for w, m in self.data])

    def __add__(self, other):
        return operator(self.data + other.data)

    def __radd__(self, other):
        if other == 0:
            return self
        return self + other

    def __sub__(self, other):
        return self + (-1) * other

    def simplified(self):
        c = dict([(m, 0) for w, m in self.data])
        for w, m in self.data:
            c[m] += w
        return operator([(c[m], m) for m in c if abs(c[m]) > 1e-14])

    def normalized(self):
        c = dict([(m, 0) for w, m in self.data])
        n2 = 0.0
        for w, m in self.data:
            c[m] += w
            n2 += abs(w) ** 2
        return operator([(c[m] / n2**0.5, m) for m in c if abs(c[m]) > 1e-14])

    def basis(self):
        return list(set([m for w, m in self.data]))

    def vector(self, basis):
        assert all(m in basis for w, m in self.data)
        c = dict([(m, 0) for w, m in self.data])
        for w, m in self.data:
            c[m] += w
        return [c[m] if m in c else 0 for m in basis]
