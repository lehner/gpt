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
from gpt.core.matrix import exp as mexp

from gpt.algorithms.integrators.molecular_dynamics import leap_frog, OMF4


class update:
    def __init__(self, first, second):
        if not type(first) is list:
            self.fld = [first]
        else:
            self.fld = first

        if not type(second) is list:
            tmp = [second]
        else:
            tmp = second

        if hasattr(tmp[0], "force"):
            self.act = tmp
        else:
            self.mom = tmp

    def get_type(self):
        if self.fld[0].otype is gpt.otype.ot_matrix_su3_fundamental:
            return 0
        elif self.fld[0].otype is gpt.otype.ot_singlet:
            return 1

    def __call__(self, eps):
        for mu in range(len(self.fld)):
            if hasattr(self, "mom"):
                if self.get_type() == 0:
                    self.fld[mu] @= mexp(gpt.eval(eps * self.mom[mu])) * self.fld[mu]
                elif self.get_type() == 1:
                    self.fld[mu] += eps * self.mom[mu]
            else:
                for a in self.act:
                    a.pre_force()
                    frc = a.force(mu)
                    self.fld[mu] -= eps * frc

    def get_act(self):
        if hasattr(self, "act"):
            return self.act
        return []
