#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Mattia Bruno
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
from gpt.core.group import differentiable_functional


class mass_term(differentiable_functional):
    def __init__(self, m=1.0):
        self.m = m
        self.__name__ = f"mass_term({m})"

    def __call__(self, pi):
        return g.group.inner_product(pi, pi) * self.m * 0.5

    @differentiable_functional.multi_field_gradient
    def gradient(self, pi, dpi):
        dS = []
        for _pi in dpi:
            i = pi.index(_pi)
            dS.append(g(self.m * pi[i]))
        return dS
