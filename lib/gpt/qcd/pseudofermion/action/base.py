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


class action_base(differentiable_functional):
    def __init__(self, M, inverter, operator):
        self.M = g.core.util.to_list(M)
        self.inverter = inverter
        self.operator = operator

    def _updated(self, fields):
        U = fields[0:-1]
        psi = fields[-1]
        return [m.updated(U) for m in self.M] + [U, psi]

    def _allocate_force(self, U):
        frc = g.group.cartesian(U)
        for f in frc:
            f[:] = 0
        return frc

    def _accumulate(self, frc, frc1, sign):
        for f, f1 in zip(frc, frc1):
            f += sign * f1

    def __call__(self, fields):
        raise NotImplementedError()

    def draw(self, fields, rng):
        raise NotImplementedError()

    def gradient(self, fields, dfields):
        raise NotImplementedError()

    def transformed(self, t):
        return transformed_action(self, t)


class transformed_action(action_base):
    def __init__(self, a, t):
        self.a = a
        self.at = differentiable_functional.transformed(a, t)
        self.t = t

    def __call__(self, fields):
        return self.at(fields)

    def gradient(self, fields, dfields):
        return self.at.gradient(fields, dfields)

    def draw(self, fields, *x):
        return self.a.draw(self.t(fields), *x)
