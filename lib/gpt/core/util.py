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
import numpy as np

# test if of number type
def isnum(x):
    return isinstance(x, (int, float, complex)) and not isinstance(x, bool)


# tensor
def value_to_tensor(val, otype):
    if otype == gpt.ot_singlet:
        return complex(val)
    return gpt.tensor(val, otype)


def tensor_to_value(value, dtype=np.complex128):
    if type(value) == gpt.tensor:
        value = value.array
        if value.dtype != dtype:
            value = dtype(value)
    elif isnum(value):
        value = np.array([value], dtype=dtype)
    return value


# list
def to_list(value):
    if type(value) == list:
        return value
    return [value]


def from_list(value):
    if type(value) == list and len(value) == 1:
        return value[0]
    return value


def is_list_instance(value, t):
    return isinstance(value, t) or (type(value) == list and isinstance(value[0], t))
