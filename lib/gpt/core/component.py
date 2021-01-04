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


imag = _simple_matrix("imag")
real = _simple_matrix("real")
abs = _simple_matrix("abs")
sqrt = _simple_matrix("sqrt")
exp = _simple_matrix("exp")
log = _simple_matrix("log")
sin = _simple_matrix("sin")
asin = _simple_matrix("asin")
cos = _simple_matrix("cos")
acos = _simple_matrix("acos")
tan = _simple_matrix("tan")
atan = _simple_matrix("atan")
inv = _simple_matrix("pow", {"exponent": -1.0})


def pow(exponent):
    return _simple_matrix("pow", {"exponent": exponent})
