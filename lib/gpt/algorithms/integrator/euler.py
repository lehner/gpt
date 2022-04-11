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

import gpt

# dx/dt = f(t,x(t))
# x(eps) = x(0) + eps * f(0, x(0))


class euler:
    def __init__(self, dst, func, sign):
        self.dst = gpt.core.util.to_list(dst)
        self.func = func
        self.sign = sign

    def __call__(self, eps):
        funcs = gpt.core.util.to_list(self.func())
        for i in range(len(funcs)):
            self.dst[i] @= gpt.group.compose(gpt.eval(self.sign * eps * funcs[i]), self.dst[i])
