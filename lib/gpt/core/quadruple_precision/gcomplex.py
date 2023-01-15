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
import numpy as np


class gcomplex:
    def __init__(self, x, y=None):
        self.real = x
        if y is None:
            y = x * 0.0
        self.imag = y

    def __iadd__(self, other):
        res = self + other
        self.real, self.imag = res.real, res.imag
        return self

    def __add__(self, other):
        if not isinstance(other, gcomplex):
            other = self.__class__(other)
        return self.__class__(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other):
        return self + other

    def __isub__(self, other):
        res = self - other
        self.real, self.imag = res.real, res.imag
        return self

    def __sub__(self, other):
        if not isinstance(other, gcomplex):
            other = self.__class__(other)
        return self.__class__(self.real - other.real, self.imag - other.imag)

    def __rsub__(self, other):
        return self.__class__(other.real - self.real, other.imag - self.imag)

    def __imul__(self, other):
        res = self * other
        self.real, self.imag = res.real, res.imag
        return self

    def __mul__(self, other):
        if not isinstance(other, gcomplex):
            other = self.__class__(other)
        return self.__class__(
            self.real * other.real - self.imag * other.imag,
            self.imag * other.real + self.real * other.imag,
        )

    def __rmul__(self, other):
        return self * other

    def __itruediv__(self, other):
        res = self / other
        self.real, self.imag = res.real, res.imag
        return self

    def __truediv__(self, other):
        if not isinstance(other, gcomplex):
            other = self.__class__(other)
        return self.__mul__(other.inv())

    def __rtruediv__(self, other):
        return self.inv().__mul__(other)

    def conjugate(self):
        return self.__class__(self.real, -self.imag)

    def __abs__(self):
        return (self.real * self.real + self.imag * self.imag).sqrt()

    def inv(self):
        n = 1.0 / (self.real * self.real + self.imag * self.imag)
        return self.__class__(n * self.real, -n * self.imag)

    def __repr__(self):
        return f"{self.real} + {self.imag}j"

    def to_serial(self):
        return np.stack([self.real.to_serial(), self.imag.to_serial()])

    def from_serial(self, serial):
        return self.__class__(self.real.from_serial(serial[0]), self.imag.from_serial(serial[1]))
