#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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
import gpt, cgpt
from gpt.qcd.fermion.operator.base import base
from gpt.qcd.fermion.operator.interface import interface


class coarse_operator(base):
    def __init__(self, name, U, params, otype=None):

        self.interface = interface()

        super().__init__(name, U, params, otype, True)

        self.tmp = gpt.lattice(self.F_grid, otype)
        self.tmp_eo = gpt.lattice(self.F_grid_eo, otype)
        self.U_self_inv = gpt.matrix.inv(self.U[8])
        self.t = gpt.timer("coarse_operator")

        self.params["U"] = [v_obj for u in U for v_obj in u.v_obj]
        self.params["U_self_inv"] = self.U_self_inv.v_obj

        self.interface.setup(name, self.U_grid, self.params)
