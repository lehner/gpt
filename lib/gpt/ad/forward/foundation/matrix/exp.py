#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt as g
import numpy as np


def function(x):
    fac = 1.0
    base = 128.0
    nbase = 7
    x = x / base
    c = x
    r = g.identity(x)
    for i in range(1, 10):
        fac /= float(i)
        r = r + fac * c
        c = c * x
    for i in range(nbase):
        r = r * r
    return r


# gives 1e-14 / 1e-15 errors up to at least |x| < 10


def function_and_gradient(x, dx):
    assert False
