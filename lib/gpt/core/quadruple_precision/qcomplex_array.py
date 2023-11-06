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
import gpt as g
from gpt.core.quadruple_precision.gcomplex import gcomplex
from gpt.core.quadruple_precision.qfloat_array import qfloat_array

HANDLED_FUNCTIONS = {}


class qcomplex_array(gcomplex, np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, x, y=None):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            x, y = qfloat_array(x.real), qfloat_array(x.imag)
        elif isinstance(x, qcomplex_array):
            x, y = x.real, x.imag
        else:
            if not isinstance(x, qfloat_array):
                x = qfloat_array(x)
            if y is not None and not isinstance(y, qfloat_array):
                y = qfloat_array(y)
        super().__init__(x, y)

    def leading(self):
        return self.real.leading() + 1j * self.imag.leading()

    def __repr__(self):
        return gcomplex.__repr__(self)

    def __array__(self, dtype=None):
        return NotImplemented

    def __getitem__(self, index):
        return gcomplex(self.real[index], self.imag[index])

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


@implements(np.imag)
def np_imag(arr):
    return arr.imag


@implements(np.real)
def np_real(arr):
    return arr.real


@implements(np.abs)
def np_abs(arr):
    return (arr.real * arr.real + arr.imag * arr.imag).sqrt()


@implements(np.linalg.norm)
def np_linalg_norm(arr):
    arr = (arr * arr.conjugate()).real
    r = g.qfloat(0.0)
    for x, y in zip(arr.x.flat, arr.y.flat):
        r += g.qfloat(x, y)
    return r.sqrt()
