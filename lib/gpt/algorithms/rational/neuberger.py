#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2022  Raphael Lehner (raphael.lehner@physik.uni-regensburg.de)
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


#
# neuberger_inverse_square_root approximates 1/sqrt(x^2) with
#         (z+u_1) (z+u_2)
#  norm * ------- ------- ...
#         (z+v_1) (z+v_2)
#
class neuberger_inverse_square_root:
    def __init__(self, low, high, order):
        m = (high + low) / 2.0
        r = (high - low) / 2.0
        assert 0.0 < r and r < m
        self.m = m
        self.r = r
        self.n = order

        c = 1.0 / ((m + r) * (m - r)) ** 0.5
        d = ((m + r) / (m - r)) ** 0.5
        e = numpy.pi / 2.0 / order
        A = numpy.sum([1.0 / numpy.cos(e * (i + 0.5)) ** 2.0 for i in range(order)])
        a = numpy.array([-((numpy.tan(e / 2.0 * i) / c) ** 2.0) for i in range(1, 2 * order)])

        self.zeros = a[1::2]
        self.poles = a[0::2]
        self.norm = A / c / order
        self.delta = 2.0 / (((d + 1.0) / ((d - 1.0))) ** (2.0 * order) - 1.0)

    def __str__(self):
        out = f"Neuberger approx of 1/sqrt(x^2) with {self.n} poles "
        out += f"in the circle C(m = {self.m}, r = {self.r})\n"
        out += f"   relative error delta = {self.delta:e}"
        return out
