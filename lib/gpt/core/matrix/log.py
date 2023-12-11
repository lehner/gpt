#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-22  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt, numpy


def log(i, convergence_threshold=0.0001):
    x = gpt.eval(i)

    if x.grid.precision is gpt.single:
        # perform intermediate calculation with enhanced precision
        return gpt.convert(log(gpt.convert(x, gpt.double)), gpt.single)

    Id = gpt.identity(x)

    nrm = gpt.norm2(Id)

    # log(x^{1/2^n}) = 1/2^n log(x)
    scale = 1.0
    for n in range(20):
        x = gpt.matrix.sqrt(x)

        eps2 = gpt.norm2(x - Id) / nrm
        scale *= 2.0
        if eps2 < convergence_threshold:
            break

    x = gpt(x - Id)
    o = gpt.copy(x)
    xn = x
    for j in range(2, 8):
        xn = gpt(xn * x)
        o -= xn * ((-1.0) ** j / j)

    o = gpt(scale * o)
    return o
