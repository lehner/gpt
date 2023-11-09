#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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
import copy


# test if of number type
def is_num(x):
    return isinstance(x, (int, float, complex, gpt.qfloat, gpt.qcomplex)) and not isinstance(
        x, bool
    )


# adj a number
def adj_num(x):
    if isinstance(x, (int, float, gpt.qfloat)):
        return x
    elif isinstance(x, complex):
        return x.conjugate()
    else:
        raise Exception(f"adj_num not yet implemented for type {type(x)}")


# convert to number type
def to_num(x):
    if isinstance(x, (np.complex64, np.complex128)):
        return complex(x)
    elif isinstance(x, (np.float32, np.float64)):
        return float(x)
    elif isinstance(x, (np.int32, np.int64)):
        return int(x)
    return x


# tensor
def value_to_tensor(val, otype):
    if otype.data_otype() == gpt.ot_singlet:
        # this is not ideal, can we do a subclass of complex that preserves otype info?
        return complex(val)
    return gpt.tensor(val, otype)


def tensor_to_value(value, dtype=np.complex128):
    if isinstance(value, gpt.tensor):
        value = value.array
        if value.dtype != dtype:
            value = dtype(value)
    elif is_num(value):
        value = np.array([value], dtype=dtype)
    return value


# list
def to_list(*values):
    if len(values) > 1:
        return zip(*tuple([to_list(v) for v in values]))
    elif len(values) == 1:
        value = values[0]
        if isinstance(value, list):
            return value
        return [value]


def from_list(value):
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def is_list_instance(value, t):
    return isinstance(value, t) or (isinstance(value, list) and isinstance(value[0], t))


def entries_have_length(value, count):
    if isinstance(value, list):
        return all([len(v) == count for v in value])


# callable
def is_callable(value):
    if isinstance(value, list):
        return all([is_callable(v) for v in value])
    return callable(value) or value is None


def all_have_attribute(value, a):
    if isinstance(value, list) and len(value) > 0:
        return all([all_have_attribute(v, a) for v in value])
    return hasattr(value, a)
