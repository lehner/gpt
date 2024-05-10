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
import gpt as g

from gpt.core.quadruple_precision.dekker_tuple import dekker_tuple

HANDLED_FUNCTIONS = {}


class qfloat_array(dekker_tuple, np.lib.mixins.NDArrayOperatorsMixin):
    _medium = float(2**27 + 1)  # 1 .. 52
    _zero = 0.0

    def _zero_to_one(x):
        m = x == 0.0
        x[m] = 1.0
        return m

    def _set_mask(x, m, v):
        x[m] = v

    def __init__(self, x, y=None):
        if not isinstance(x, (int, float, np.ndarray, list, np.float64, np.int64)):
            raise NotImplementedError()
        x = np.array(x, dtype=np.float64)
        if y is None:
            y = np.zeros(shape=x.shape, dtype=np.float64)
        dekker_tuple.__init__(self, x, y)

    def __repr__(self):
        return f"({self.x} + {self.y})"

    def __array__(self, dtype=None):
        return NotImplemented

    def __getitem__(self, index):
        return g.qfloat(self.x[index], self.y[index])

    def __float__(self):
        if self.x.shape != (1,):
            return float("nan")
        return float(self.x[0])

    def to_serial(self):
        return np.stack([self.x, self.y])

    def from_serial(self, serial):
        return self.__class__(serial[0], serial[1])

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc not in HANDLED_FUNCTIONS or method != "__call__":
            return NotImplemented
        return HANDLED_FUNCTIONS[ufunc](*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def implements(np_function):
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@implements(np.abs)
def np_abs(arr):
    return (arr * arr).sqrt()


@implements(np.sqrt)
def np_sqrt(arr):
    return arr.sqrt()


@implements(np.real)
def np_real(arr):
    return arr


@implements(np.imag)
def np_imag(arr):
    return 0.0 * arr


@implements(np.multiply)
def np_multiply(a, b):
    return b * a


@implements(np.linalg.norm)
def np_linalg_norm(arr):
    r = g.qfloat(0.0)
    for x, y in zip(arr.x.flat, arr.y.flat):
        z = g.qfloat(x, y)
        r += z * z
    return r.sqrt()
