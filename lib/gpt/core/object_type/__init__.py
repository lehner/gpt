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
from gpt.core.object_type.base import *
from gpt.core.object_type.container import *
from gpt.core.object_type.su_n import *
import numpy

###
# Helper to create lattice / tensor of specific type
def gpt_object(first, ot):
    if type(first) == gpt.grid:
        return gpt.lattice(first, ot)
    return gpt.tensor(numpy.array(first, dtype=numpy.complex128), ot)


###
# Container objects without lie group structure
def singlet(grid):
    return gpt_object(grid, ot_singlet)


def matrix_color(grid, ndim):
    return gpt_object(grid, ot_matrix_color(ndim))


def vector_color(grid, ndim):
    return gpt_object(grid, ot_vector_color(ndim))


def matrix_spin(grid, ndim):
    return gpt_object(grid, ot_matrix_spin(ndim))


def vector_spin(grid, ndim):
    return gpt_object(grid, ot_vector_spin(ndim))


def matrix_spin_color(grid, spin_ndim, color_ndim):
    return gpt_object(grid, ot_matrix_spin_color(spin_ndim, color_ndim))


def vector_spin_color(grid, spin_ndim, color_ndim):
    return gpt_object(grid, ot_vector_spin_color(spin_ndim, color_ndim))


def vector_singlet(grid, n):
    return gpt_object(grid, ot_vector_singlet(n))


def matrix_singlet(grid, n):
    return gpt_object(grid, ot_matrix_singlet(n))


###
# Container objects with lie group structure
def matrix_su2_fundamental(grid):
    return gpt_object(grid, ot_matrix_su_n_fundamental_group(2))


def matrix_su2_adjoint(grid):
    return gpt_object(grid, ot_matrix_su_n_adjoint_group(2))


def matrix_su3_fundamental(grid):
    return gpt_object(grid, ot_matrix_su_n_fundamental_group(3))


###
# String conversion for safe file input
def str_to_otype(s):

    # first parse string
    a = s.split("(")
    if len(a) == 2:
        assert a[1][-1] == ")"
        root = a[0]
        # convert through int to avoid possibility of malicous code being executed in eval below
        args = "(%s)" % (
            ",".join(
                [str(int(x)) for x in filter(lambda x: x != "", a[1][:-1].split(","))]
            )
        )
    else:
        root = a[0]
        args = ""

    # then map to type
    known_types = set(
        [
            "ot_singlet",
            "ot_matrix_spin",
            "ot_vector_spin",
            "ot_matrix_color",
            "ot_vector_color",
            "ot_matrix_spin_color",
            "ot_vector_spin_color",
            "ot_matrix_su_n_fundamental_group",
            "ot_matrix_su_n_fundamental_algebra",
            "ot_matrix_su_n_adjoint_group",
            "ot_matrix_su_n_adjoint_algebra",
            "ot_vector_singlet",
            "ot_vector_singlet4",
            "ot_vector_singlet10",
            "ot_vector_singlet60",
            "ot_matrix_singlet",
            "ot_matrix_singlet4",
            "ot_matrix_singlet10",
            "ot_matrix_singlet60",
        ]
    )

    assert root in known_types
    return eval(root + args)


###
# aliases
def complex(grid):
    return singlet(grid)


def vcomplex(grid, n):
    return vector_singlet(grid, n)


def mcomplex(grid, n):
    return matrix_singlet(grid, n)


def mcolor(grid):
    return matrix_su3_fundamental(grid)


def vcolor(grid):
    return vector_color(grid, 3)


def mspin(grid):
    return matrix_spin(grid, 4)


def vspin(grid):
    return vector_spin(grid, 4)


def mspincolor(grid):
    return matrix_spin_color(grid, 4, 3)


def vspincolor(grid):
    return vector_spin_color(grid, 4, 3)
