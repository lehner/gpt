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
import gpt, sys
import numpy

# need a basic container
from gpt.core.object_type import ot_singlet

###
# (\mathbb{R},+)
class ot_real_additive_group(ot_singlet):
    Ndim = 1

    def __init__(self):
        self.__name__ = "ot_real_additive_group"
        self.data_alias = lambda: ot_singlet
        self.rmtab = {
            "ot_singlet": (lambda: ot_singlet, None),
        }
        self.mtab = {
            self.__name__: (lambda: self, None),
            "ot_singlet": (lambda: ot_singlet, None),
        }

    # this is always multiplicative identity, not neutral element of group
    def identity(self):
        return 1.0

    def compose(self, a, b):
        return a + b

    def cartesian(self):
        return self

    def generators(self, dt):
        return [1.0]

    def coordinates(self, l, c=None):
        if c is None:
            return [gpt.component.real(l)]
        else:
            l @= c[0]

    def is_element(self, U):
        err2 = gpt.norm2(gpt.component.imag(U)) / U.grid.gsites
        return err2 ** 0.5 < U.grid.precision.eps * 10.0

    def project(self, U, method):
        # there is really only one sensible projection, ignore method
        U @= gpt.component.real(U)
