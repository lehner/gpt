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
from gpt.core.operator.matrix_operator import matrix_operator
import cgpt


def _simple_matrix(operator, extra_params={}):
    def _mat(dst, src):
        for i in dst.otype.v_idx:
            cgpt.unary(
                dst.v_obj[i], src.v_obj[i], {**{"operator": operator}, **extra_params}
            )

    return matrix_operator(_mat)


def _simple_map(operator, extra_params={}):
    def _mat(first, second=None):
        if second is not None:
            dst = first
            src = gpt.eval(second)
        else:
            src = gpt.eval(first)
            dst = gpt.lattice(src)
        for i in dst.otype.v_idx:
            cgpt.unary(
                dst.v_obj[i], src.v_obj[i], {**{"operator": operator}, **extra_params}
            )
        return dst

    return _mat


imag = _simple_matrix("imag")
real = _simple_matrix("real")
abs = _simple_map("abs")
sqrt = _simple_map("sqrt")
exp = _simple_map("exp")
log = _simple_map("log")
sin = _simple_map("sin")
asin = _simple_map("asin")
cos = _simple_map("cos")
acos = _simple_map("acos")
tan = _simple_map("tan")
atan = _simple_map("atan")
sinh = _simple_map("sinh")
asinh = _simple_map("asinh")
cosh = _simple_map("cosh")
acosh = _simple_map("acosh")
tanh = _simple_map("tanh")
atanh = _simple_map("atanh")
inv = _simple_map("pow", {"exponent": -1.0})


def pow(exponent):
    return _simple_map("pow", {"exponent": exponent})
