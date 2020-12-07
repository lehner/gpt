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

# need a basic container to store the group/algebra data
from gpt.core.object_type import ot_matrix_color, ot_vector_color

###
# TODO:
# first generalize the below to general SU(n) fundamental and adjoint; distinguish group and algebra fields
# allow for conversion (fundamental <> adjoint is already implemented in representation.py, can merge?) -> type conversion table, then use gpt.convert !!
#


###
# Representations of groups
class ot_matrix_su_n_fundamental_algebra(ot_matrix_color):
    def __init__(self, Nc):
        self.Nc = Nc
        super().__init__(Nc)
        self.__name__ = f"ot_matrix_su_n_fundamental_algebra({Nc})"
        self.data_alias = lambda: ot_matrix_color(Nc)
        self.mtab = {
            self.__name__: (lambda: self, (1, 0)),
            f"ot_vector_color({Nc})": (lambda: ot_vector_color(Nc), (1, 0)),
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }


class ot_matrix_su_n_fundamental_group(ot_matrix_color):
    def __init__(self, Nc):
        self.Nc = Nc
        super().__init__(Nc)  # need 3 dim lattice
        self.__name__ = f"ot_matrix_su_n_fundamental_group({Nc})"
        self.data_alias = lambda: ot_matrix_color(Nc)
        self.mtab = {
            self.__name__: (lambda: self, (1, 0)),
            f"ot_vector_color({Nc})": (lambda: ot_vector_color(Nc), (1, 0)),
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }

    def generators(self, dt):
        r = []

        # accumulate generators in pauli / Gell-Mann ordering
        for i in range(self.Nc):
            for j in range(i + 1, self.Nc):

                # sigma_x
                alg = numpy.zeros(shape=(self.Nc, self.Nc), dtype=dt)
                alg[i, j] = 1.0
                alg[j, i] = 1.0
                r.append(alg)

                # sigma_y
                alg = numpy.zeros(shape=(self.Nc, self.Nc), dtype=dt)
                alg[i, j] = -1j
                alg[j, i] = 1j
                r.append(alg)

                # sigma_z
                if j == i + 1:
                    alg = numpy.zeros(shape=(self.Nc, self.Nc), dtype=dt)
                    for l in range(j):
                        alg[l, l] = 1.0
                    alg[j, j] = -j
                    r.append(alg)

        # need to satisfy normalization Tr(T_a T_b) = 1/2 delta_{ab}
        for alg in r:
            alg /= (numpy.trace(numpy.dot(alg, alg)) * 2.0) ** 0.5

        # return gpt_object version
        algebra_otype = ot_matrix_su_n_fundamental_algebra(self.Nc)
        return [gpt.gpt_object(i, algebra_otype) for i in r]


class ot_matrix_su_n_adjoint_algebra(ot_matrix_color):
    def __init__(self, Nc):
        self.Nc = Nc
        self.Ndim = Nc * Nc - 1
        super().__init__(self.Ndim)
        self.__name__ = f"ot_matrix_su_n_adjoint_algebra({Nc})"
        self.data_alias = lambda: ot_matrix_color(self.Ndim)
        self.mtab = {
            self.__name__: (lambda: self, (1, 0)),
            f"ot_vector_color({self.Ndim})": (
                lambda: ot_vector_color(self.Ndim),
                (1, 0),
            ),
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }


class ot_matrix_su_n_adjoint_group(ot_matrix_color):
    def __init__(self, Nc):
        self.Nc = Nc
        self.Ndim = Nc * Nc - 1
        super().__init__(self.Ndim)
        self.__name__ = f"ot_matrix_su_n_adjoint_group({Nc})"
        self.data_alias = lambda: ot_matrix_color(self.Ndim)
        self.mtab = {
            self.__name__: (lambda: self, (1, 0)),
            f"ot_vector_color({self.Ndim})": (
                lambda: ot_vector_color(self.Ndim),
                (1, 0),
            ),
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }

    def generators(self, dt):
        T_f = ot_matrix_su_n_fundamental_group(self.Nc).generators(dt)
        f = [
            [
                [
                    numpy.trace(
                        (T_f[a].array @ T_f[b].array - T_f[b].array @ T_f[a].array)
                        @ T_f[c].array
                    )
                    * 2.0
                    / 1j
                    for c in range(self.Ndim)
                ]
                for b in range(self.Ndim)
            ]
            for a in range(self.Ndim)
        ]
        r = []
        for a in range(self.Ndim):
            arr = numpy.array(f[a], dtype=dt) / 1j
            r.append(arr)

        # need to satisfy normalization Tr(T_a T_b) = 1/2 delta_{ab}
        for alg in r:
            alg /= (numpy.trace(numpy.dot(alg, alg)) * 2.0) ** 0.5

        # return gpt_object version
        algebra_otype = ot_matrix_su_n_adjoint_algebra(self.Nc)
        return [gpt.gpt_object(i, algebra_otype) for i in r]
