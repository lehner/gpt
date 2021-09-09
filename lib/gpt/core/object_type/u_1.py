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
# U1
class ot_u_1_base(ot_singlet):
    Nc = 1
    Ndim = 1

    def identity(self):
        return complex(1.0, 0.0)

    def __init__(self, name):
        self.__name__ = name
        self.data_alias = lambda: ot_singlet
        self.rmtab = {
            "ot_singlet": (lambda: ot_singlet, None),
        }
        self.mtab = {
            self.__name__: (lambda: self, None),
            "ot_singlet": (lambda: ot_singlet, None),
        }


class ot_u_1_algebra(ot_u_1_base):
    def __init__(self):
        super().__init__("ot_u_1_algebra")
        self.ctab = {
            "ot_u_1_group": lambda dst, src: gpt.eval(dst, gpt.component.exp(src * 1j))
        }

    def cartesian(self):
        return self

    def compose(self, a, b):
        return a + b

    def generators(self, dt):
        return [complex(1.0, 0)]

    def inner_product(self, left, right):
        return gpt.inner_product(left, right).real

    def coordinates(self, l, c=None):
        if c is None:
            return [l]
        else:
            l @= c


class ot_u_1_group(ot_u_1_base):
    def __init__(self):
        super().__init__("ot_u_1_group")
        self.ctab = {
            "ot_u_1_algebra": lambda dst, src: gpt.eval(
                dst, gpt.component.log(src) / 1j
            )
        }

    def compose(self, a, b):
        return a * b

    def defect(self, U):
        I = gpt.identity(U)
        err2 = gpt.norm2(U * gpt.adj(U) - I) / gpt.norm2(I)
        return err2 ** 0.5

    def project(self, U, method):
        if method == "defect_right" or method == "defect":
            I = gpt.identity(U)
            eps = gpt.eval(0.5 * gpt.adj(U) * U - 0.5 * I)
            U @= U * (I - eps)
        elif method == "defect_left":
            I = gpt.identity(U)
            eps = gpt.eval(0.5 * U * gpt.adj(U) - 0.5 * I)
            U @= (I - eps) * U
        else:
            raise Exception("Unknown projection method")

    def cartesian(self):
        return ot_u_1_algebra()
