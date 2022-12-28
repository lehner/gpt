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
#
#    Original Idea:  Dekker, T. J. Numerische Mathematik 18 (1971) 224-242
#
import numpy as np


def _dekker_add_one_eps(x, y):
    # project to standard separation of lead and correction
    l = x + y
    c = y - (l - x)
    return l, c


def _dekker_add(x, y):
    l = x + y
    t = l - x
    c = (x - (l - t)) + (y - t)
    return l, c


def _dekker_sub(x, y):
    l = x - y
    t = l - x
    c = (x - (l - t)) - (y + t)
    return l, c


def _dekker_mul(x, y, medium):
    u = x * medium
    v = y * medium
    s = u - (u - x)
    t = v - (v - y)
    f = x - s
    g = y - t
    l = x * y
    c = ((s * t - l) + s * g + f * t) + f * g
    return l, c


class dekker_tuple:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def normalize(self):
        self.x, self.y = _dekker_add_one_eps(self.x, self.y)
        return self

    def __eq__(self, other):
        if not isinstance(other, dekker_tuple):
            other = self.__class__(other)

        return np.logical_and(self.x == other.x, self.y == other.y)

    def __le__(self, other):
        if not isinstance(other, dekker_tuple):
            other = self.__class__(other)

        return self.x <= other.x

    def __neg__(self):
        return self.__class__(-self.x, -self.y)

    def __iadd__(self, other):
        res = self + other
        self.x, self.y = res.x, res.y
        return self

    def __add__(self, other):
        if not isinstance(other, dekker_tuple):
            other = self.__class__(other)
        l, c = _dekker_add(self.x, other.x)
        c += self.y + other.y
        return self.__class__(l, c).normalize()

    def __radd__(self, other):
        return self.__class__(other) + self

    def __isub__(self, other):
        res = self - other
        self.x, self.y = res.x, res.y
        return self

    def __sub__(self, other):
        if not isinstance(other, dekker_tuple):
            other = self.__class__(other)
        l, c = _dekker_sub(self.x, other.x)
        c += self.y - other.y
        return self.__class__(l, c).normalize()

    def __rsub__(self, other):
        return self.__class__(other) - self

    def __imul__(self, other):
        res = self * other
        self.x, self.y = res.x, res.y
        return self

    def __mul__(self, other):
        if not isinstance(other, dekker_tuple):
            try:
                other = self.__class__(other)
            except NotImplementedError:
                return other.__rmul__(self)
        l, c = _dekker_mul(self.x, other.x, self.__class__._medium)
        c += self.x * other.y + self.y * other.x
        return self.__class__(l, c).normalize()

    def __rmul__(self, other):
        return self.__class__(other) * self

    def __itruediv__(self, other):
        res = self / other
        self.x, self.y = res.x, res.y
        return self

    def __truediv__(self, other):
        if not isinstance(other, dekker_tuple):
            other = self.__class__(other)
        l = self.x / other.x
        s, f = _dekker_mul(l, other.x, self.__class__._medium)
        c = (self.x - s - f + self.y - l * other.y) / other.x
        return self.__class__(l, c).normalize()

    def __rtruediv__(self, other):
        return self.__class__(other) / self

    def sqrt(self):
        c = self.__class__
        r = self.x**0.5
        m = c._zero_to_one(r)
        s, f = _dekker_mul(r, r, c._medium)
        e = (self.x - s - f + self.y) * 0.5 / r
        c._set_mask(r, m, c._zero)
        c._set_mask(e, m, c._zero)
        return c(r, e).normalize()

    def leading(self):
        return self.x

    def __repr__(self):
        return f"dekker_tuple({self.x},{self.y})"
