#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020 Tilo Wettig
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
from gpt.core.object_type import ot_matrix_color, ot_vector_color

###
# Representations of groups
class ot_matrix_su3_fundamental(ot_matrix_color):
    def __init__(self):
        self.Nc = 3
        super().__init__(3)  # need 3 dim lattice
        self.__name__ = "ot_matrix_su3_fundamental()"
        self.data_alias = lambda: ot_matrix_color(3)
        self.mtab = {
            self.__name__: (lambda: self, (1, 0)),
            "ot_vector_color(3)": (lambda: ot_vector_color(3), (1, 0)),
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }

    def generators(self, dt):
        # Generators always need to satisfy normalization Tr(T_a T_b) = 1/2 delta_{ab}
        return [
            gpt.matrix_su3_fundamental(i)
            for i in [
                numpy.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=dt) / 2.0,
                numpy.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=dt) / 2.0,
                numpy.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=dt) / 2.0,
                numpy.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=dt) / 2.0,
                numpy.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=dt) / 2.0,
                numpy.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=dt) / 2.0,
                numpy.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=dt) / 2.0,
                numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=dt)
                / (3.0) ** 0.5
                / 2.0,
            ]
        ]


class ot_matrix_su2_fundamental(ot_matrix_color):
    def __init__(self):
        self.Nc = 2
        super().__init__(2)  # need 2 dim matrices
        self.__name__ = "ot_matrix_su2_fundamental()"
        self.data_alias = lambda: ot_matrix_color(2)
        self.mtab = {
            self.__name__: (lambda: self, (1, 0)),
            "ot_vector_color(2)": (lambda: ot_vector_color(2), (1, 0)),
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }

    def generators(self, dt):
        # The generators are normalized such that T_a^2 = Id/2Nc + d_{aab}T_b/2
        # Generators always need to satisfy normalization Tr(T_a T_b) = 1/2 delta_{ab}
        return [
            gpt.matrix_su2_fundamental(i)
            for i in [
                numpy.array([[0, 1], [1, 0]], dtype=dt) / 2.0,
                numpy.array([[0, -1j], [1j, 0]], dtype=dt) / 2.0,
                numpy.array([[1, 0], [0, -1]], dtype=dt) / 2.0,
            ]
        ]


class ot_matrix_su2_adjoint(ot_matrix_color):
    def __init__(self):
        self.Nc = 2
        super().__init__(3)  # need 3 dim matrices
        self.__name__ = "ot_matrix_su2_adjoint()"
        self.data_alias = lambda: ot_matrix_color(3)
        self.mtab = {
            self.__name__: (lambda: self, (1, 0)),
            "ot_vector_color(3)": (lambda: ot_vector_color(3), (1, 0)),
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }

    def generators(self, dt):
        # (T_i)_{kj} = c^k_{ij} with c^k_{ij} = i \epsilon_{ijk} / 2
        # Generators always need to satisfy normalization Tr(T_a T_b) = 1/2 delta_{ab}
        return [
            gpt.matrix_su2_adjoint(i)
            for i in [
                numpy.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=dt) / 2.0,
                numpy.array([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]], dtype=dt) / 2.0,
                numpy.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=dt) / 2.0,
            ]
        ]
