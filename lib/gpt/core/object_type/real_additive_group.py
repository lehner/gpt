#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt
import numpy

# need a basic container
from gpt.core.object_type import ot_singlet, ot_matrix_singlet, ot_vector_singlet

###
# (\mathbb{R},+)
class ot_real_additive_group(ot_singlet):
    Ndim = 1

    def __init__(self):
        self.__name__ = "ot_real_additive_group"
        self.data_alias = lambda: ot_singlet
        self.rmtab = {
            "ot_singlet": (lambda: ot_singlet, None),
        }
        self.mtab = {
            self.__name__: (lambda: self, None),
            "ot_singlet": (lambda: ot_singlet, None),
        }

    # this is always multiplicative identity, not neutral element of group
    def identity(self):
        return 1.0

    def compose(self, a, b):
        return a + b

    def cartesian(self):
        return self

    def generators(self, dt):
        return [1.0]

    def inner_product(self, left, right):
        return gpt.inner_product(left, right).real

    def coordinates(self, l, c=None):
        if c is None:
            return [gpt.component.real(l)]
        else:
            l @= c[0]

    def defect(self, U):
        err2 = gpt.norm2(gpt.component.imag(U)) / U.grid.gsites
        return err2 ** 0.5

    def project(self, U, method):
        # there is really only one sensible projection, ignore method
        U @= gpt.component.real(U)


###
# (\mathbb{R}^{n},+)
class ot_vector_real_additive_group(ot_vector_singlet):
    def __init__(self, n):
        super().__init__(n)
        self.__name__ = f"ot_vector_real_additive_group({n})"
        self.data_alias = lambda: ot_vector_singlet(n)
        self.mtab = {
            "ot_singlet": (lambda: self, None),
            "ot_real_additive_group": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
            "ot_real_additive_group": (lambda: self, None),
        }
        self.otab = {self.__name__: (lambda: ot_matrix_real_additive_group(n), [])}
        self.itab = {
            self.__name__: (lambda: ot_singlet, (0, 0)),
        }
        self.cache = {}

    def compose(self, a, b):
        return a + b

    def cartesian(self):
        return self

    def generators(self, dt):
        n = self.shape[0]

        def basis(i):
            m = numpy.zeros(self.shape, dtype=dt)
            m[i] = 1.0
            return gpt.vector_singlet(m, n)

        return [basis(i) for i in range(n)]

    def inner_product(self, left, right):
        return gpt.inner_product(left, right).real

    def coordinates(self, l, c=None):
        assert l.otype.__name__ == self.__name__
        if c is None:
            r = [None] * self.shape[0]
            a = gpt.separate_indices(l, (0, lambda: ot_singlet), self.cache)
            for i in a:
                r[i[0]] = a[i]
            return r
        else:
            l[:] = 0
            for ca, Ta in zip(c, self.generators(l.grid.precision.complex_dtype)):
                l += ca * Ta

    def defect(self, U):
        err2 = gpt.norm2(gpt.component.imag(U)) / U.grid.gsites / self.nfloats
        return err2 ** 0.5

    def project(self, U, method):
        # there is really only one sensible projection, ignore method
        U @= gpt.component.real(U)


###
# (\mathbb{R}^{n \times n},+)
class ot_matrix_real_additive_group(ot_matrix_singlet):
    def __init__(self, n):
        self.Ndim = n
        super().__init__(n)
        self.__name__ = f"ot_matrix_real_additive_group({n})"
        self.data_alias = lambda: ot_matrix_singlet(n)
        self.mtab = {
            self.__name__: (lambda: self, (1, 0)),
            f"ot_vector_real_additive_group({n})": (
                lambda: ot_vector_real_additive_group(n),
                (1, 0),
            ),
            "ot_singlet": (lambda: self, None),
            "ot_real_additive_group": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
            "ot_real_additive_group": (lambda: self, None),
        }

    def compose(self, a, b):
        return a + b

    def cartesian(self):
        return self

    def generators(self, dt):
        n = self.shape[0]

        def basis(i, j):
            m = numpy.zeros(self.shape, dtype=dt)
            m[i, j] = 1.0
            return gpt.matrix_singlet(m, n)

        return [basis(i, j) for i in range(n) for j in range(n)]

    def inner_product(self, left, right):
        return gpt.sum(gpt.trace(gpt.adj(left) * right)).real

    def coordinates(self, l, c=None):
        assert l.otype.__name__ == self.__name__
        gen = self.generators(l.grid.precision.complex_dtype)
        if c is None:
            return [gpt.eval(gpt.trace(gpt.adj(l) * Ta)) for Ta in gen]
        else:
            l[:] = 0
            for ca, Ta in zip(c, gen):
                l += ca * Ta

    def defect(self, U):
        err2 = gpt.norm2(gpt.component.imag(U)) / U.grid.gsites / self.nfloats
        return err2 ** 0.5

    def project(self, U, method):
        # there is really only one sensible projection, ignore method
        U @= gpt.component.real(U)
