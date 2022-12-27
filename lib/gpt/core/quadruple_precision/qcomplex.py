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
from gpt.core.quadruple_precision.gcomplex import gcomplex
from gpt.core.quadruple_precision.qfloat import qfloat


class qcomplex(gcomplex):
    def __init__(self, x, y=None):
        if isinstance(x, complex):
            x, y = qfloat(x.real), qfloat(x.imag)
        elif isinstance(x, qcomplex):
            x, y = x.real, x.imag
        else:
            if not isinstance(x, qfloat):
                x = qfloat(x)
            if y is not None and not isinstance(y, qfloat):
                y = qfloat(y)
        super().__init__(x, y)

    def leading(self):
        return complex(self.real.leading(), self.imag.leading())
