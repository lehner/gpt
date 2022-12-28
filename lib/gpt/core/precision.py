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
import numpy
from gpt.core.quadruple_precision import global_sum_quadruple
from gpt.core.grid import global_sum_default


class precision:
    pass


class single(precision):
    nbytes = 4
    cgpt_dtype = "single"
    real_dtype = numpy.float32
    complex_dtype = numpy.complex64
    eps = 1e-7
    global_sum_policy = global_sum_default

    def __init__(self):
        pass


class double(precision):
    nbytes = 8
    cgpt_dtype = "double"
    real_dtype = numpy.float64
    complex_dtype = numpy.complex128
    eps = 1e-15
    global_sum_policy = global_sum_default

    def __init__(self):
        pass


class double_quadruple(precision):
    nbytes = 8
    cgpt_dtype = "double"
    real_dtype = numpy.float64
    complex_dtype = numpy.complex128
    eps = 1e-15
    global_sum_policy = global_sum_quadruple

    def __init__(self):
        pass


def str_to_precision(s):
    if s == "single":
        return single
    elif s == "double":
        return double
    elif s == "double_quadruple":
        return double_quadruple
    else:
        assert 0
