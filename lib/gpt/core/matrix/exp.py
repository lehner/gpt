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
import gpt as g


class functor:
    def __init__(self):
        self.verbose = g.default.is_verbose("exp_performance")

    def __call__(self, i):

        if isinstance(i, g.expr):
            i = g(i)

        t = g.timer("exp")
        t("matrix")
        r = i.__class__.foundation.matrix.exp.function(i)
        t()
        if self.verbose:
            g.message(t)
        return r

    def function_and_gradient(self, x, dx):

        if isinstance(x, g.expr):
            x = g(x)

        return x.__class__.foundation.matrix.exp.function_and_gradient(x, dx)


exp = functor()
