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
import gpt as g


class expression:
    def __init__(self, indices, contract, evaluate):
        self.indices = indices
        self.contract = contract
        self.evaluate = evaluate

    def __rmul__(self, other):
        if g.util.is_num(other):
            return self.__mul__(other)
        return other.__mul__(self)

    def __mul__(self, other):
        if g.util.is_num(other):

            def _neval(context, path):
                return other * self.evaluate(context, path + "/#")

            def _ncontract(context, path):
                self.contract(context, path + "/#")

            return expression(self.indices, _ncontract, _neval)

        def _eval(context, path):
            f0 = self.evaluate(context, path + "/0")
            f1 = other.evaluate(context, path + "/1")
            return f0 * f1

        def _contract(context, path):
            self.contract(context, path + "/0")
            other.contract(context, path + "/1")

        return expression(list(set(self.indices + other.indices)), _contract, _eval)

    def __neg__(self):
        return self.__mul__(-1)

    def __sub__(self, other):
        return self.__add__((-1) * other)

    def __add__(self, other):
        def _eval(context, path):
            # TODO: something missing here
            assert False
            f0 = self.evaluate(context, path + "/+0")
            f1 = other.evaluate(context, path + "/+1")
            return f0 + f1

        def _contract(context, path):
            context2 = context.clone()
            self.contract(context, path + "/+0")
            other.contract(context2, path + "/+1")
            context.merge(context2)

        return expression(list(set(self.indices + other.indices)), _contract, _eval)
