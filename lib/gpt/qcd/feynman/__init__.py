#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.qcd.feynman.expression import expression
from gpt.qcd.feynman.diagrams import diagrams
from gpt.qcd.feynman.contract import contract
from gpt.qcd.feynman.draw import draw
import gpt.qcd.feynman.isospin
import numpy as np


def field(name, pos):
    return expression([(1, 0, [(name, pos)])])


e = expression([(1, 1, [])])
one = expression([(1, 0, [])])


def is_nonzero(x):
    # abstract is_nonzero allows for use of sympy for coefficients
    try:
        if abs(x) > 1e-13:
            return True
    except TypeError:
        return True
    return False


def simplify_coefficient(x):
    if isinstance(x, (int, float, complex)):
        n = float(np.round(x.real))
        if abs(n - x) < 1e-13:
            return n

        n = 1j * float(np.round(x.imag))
        if abs(n - x) < 1e-13:
            return n

        return x
    else:
        # sympy compatibility
        return x.factor().simplify()
