#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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

###
# Helper
def decompose(n, ns, rank):
    for x in reversed(sorted(ns)):
        if n % x == 0:
            return [x] * ((n // x) ** rank)
    raise Exception("Cannot decompose %d in available fundamentals %s" % (n, ns))


def get_range(ns, rank):
    # TODO: use rank to properly create rank == 2 mapping
    # This is only relevant for lattice.__str__ of mcomplex
    n = 0
    n0 = []
    n1 = []
    for x in ns:
        n0.append(n)
        n += x
        n1.append(n)
    return n0, n1


def gpt_object(first, ot):
    if type(first) == gpt.grid:
        return gpt.lattice(first, ot)
    return gpt.tensor(numpy.array(first, dtype=numpy.complex128), ot)


###
# Types below
class ot_base:
    v_otype = [None]  # cgpt's data types
    v_n0 = [0]
    v_n1 = [1]
    v_idx = [0]
    transposed = None
    spintrace = None
    colortrace = None
    data_alias = None  # ot can be cast as fundamental type data_alias (such as SU(3) -> 3x3 matrix)
    mtab = {}  # x's multiplication table for x * y
    rmtab = {}  # y's multiplication table for x * y

    # only vectors shall define otab/itab
    otab = None  # x's outer product multiplication table for x * adj(y)
    itab = None  # x's inner product multiplication table for adj(x) * y


###
# Singlet
class ot_singlet(ot_base):
    nfloats = 2
    shape = (1,)
    spintrace = (None, None, None)  # do nothing
    colortrace = (None, None, None)
    v_otype = ["ot_singlet"]
    mtab = {
        "ot_singlet": (lambda: ot_singlet, None),
    }


def singlet(grid):
    return gpt_object(grid, ot_singlet)


###
# Matrices and vectors in color space
class ot_matrix_color(ot_base):
    def __init__(self, ndim):
        self.__name__ = "ot_matrix_color(%d)" % ndim
        self.nfloats = 2 * ndim * ndim
        self.shape = (ndim, ndim)
        self.transposed = (1, 0)
        self.spintrace = (None, None, None)  # do nothing
        self.colortrace = (0, 1, lambda: ot_singlet)
        self.v_otype = ["ot_mcolor%d" % ndim]  # cgpt data types
        self.mtab = {
            self.__name__: (lambda: self, (1, 0)),
            "ot_vector_color(%d)" % ndim: (lambda: ot_vector_color(ndim), (1, 0)),
            "ot_singlet": (lambda: self, None),
        }


def matrix_color(grid, ndim):
    return gpt_object(grid, ot_matrix_color(ndim))


class ot_vector_color(ot_base):
    def __init__(self, ndim):
        self.__name__ = "ot_vector_color(%d)" % ndim
        self.nfloats = 2 * ndim
        self.shape = (ndim,)
        self.v_otype = ["ot_vcolor%d" % ndim]
        self.spintrace = (None, None)
        self.colortrace = (0, lambda: ot_singlet)
        self.mtab = {
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }
        self.otab = {self.__name__: (lambda: ot_matrix_color(ndim), [])}
        self.itab = {
            self.__name__: (lambda: ot_singlet, (0, 0)),
        }


def vector_color(grid, ndim):
    return gpt_object(grid, ot_vector_color(ndim))


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

    def generators(self, dt):
        return [
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


def matrix_su3_fundamental(grid):
    return gpt_object(grid, ot_matrix_su3_fundamental())


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

    def generators(self, dt):
        # The generators are normalized such that T_a^2 = Id/2Nc + d_{aab}T_b/2
        return [
            numpy.array([[0, 1], [1, 0]], dtype=dt) / 2.0,
            numpy.array([[0, -1j], [1j, 0]], dtype=dt) / 2.0,
            numpy.array([[1, 0], [0, -1]], dtype=dt) / 2.0,
        ]


def matrix_su2_fundamental(grid):
    return gpt_object(grid, ot_matrix_su2_fundamental())


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

    def generators(self, dt):
        # (T_i)_{kj} = c^k_{ij} with c^k_{ij} = i \epsilon_{ijk}
        return [
            numpy.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=dt),
            numpy.array([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]], dtype=dt),
            numpy.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=dt),
        ]


def matrix_su2_adjoint(grid):
    return gpt_object(grid, ot_matrix_su2_adjoint())


###
# Matrices and vectors of spin
class ot_matrix_spin(ot_base):
    def __init__(self, ndim):
        self.__name__ = "ot_matrix_spin(%d)" % ndim
        self.nfloats = 2 * ndim * ndim
        self.shape = (ndim, ndim)
        self.transposed = (1, 0)
        self.spintrace = (0, 1, lambda: ot_singlet)
        self.colortrace = (None, None, None)  # do nothing
        self.v_otype = ["ot_mspin%d" % ndim]
        self.mtab = {
            self.__name__: (lambda: self, (1, 0)),
            "ot_vector_spin(%d)" % ndim: (lambda: ot_vector_spin(ndim), (1, 0)),
            "ot_singlet": (lambda: self, None),
        }


def matrix_spin(grid, ndim):
    return gpt_object(grid, ot_matrix_spin(ndim))


class ot_vector_spin(ot_base):
    def __init__(self, ndim):
        self.__name__ = "ot_vector_spin(%d)" % ndim
        self.nfloats = 2 * ndim
        self.shape = (ndim,)
        self.v_otype = ["ot_vspin%d" % ndim]
        self.spintrace = (0, lambda: ot_singlet)
        self.colortrace = (None, None)
        self.mtab = {
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }
        self.otab = {self.__name__: (lambda: ot_matrix_spin(ndim), [])}
        self.itab = {self.__name__: (lambda: ot_singlet, (0, 0))}


def vector_spin(grid, ndim):
    return gpt_object(grid, ot_vector_spin(ndim))


###
# Matrices and vectors of both spin and color
class ot_matrix_spin_color(ot_base):
    def __init__(self, spin_ndim, color_ndim):
        self.__name__ = "ot_matrix_spin_color(%d,%d)" % (spin_ndim, color_ndim)
        self.nfloats = 2 * color_ndim * color_ndim * spin_ndim * spin_ndim
        self.shape = (spin_ndim, spin_ndim, color_ndim, color_ndim)
        self.transposed = (1, 0, 3, 2)
        self.spintrace = (0, 1, lambda: ot_matrix_color(color_ndim))
        self.colortrace = (2, 3, lambda: ot_matrix_spin(spin_ndim))
        self.v_otype = ["ot_mspin%dcolor%d" % (spin_ndim, color_ndim)]
        self.mtab = {
            self.__name__: (lambda: self, ([1, 3], [0, 2])),
            "ot_vector_spin_color(%d,%d)"
            % (spin_ndim, color_ndim): (
                lambda: ot_vector_spin(spin_ndim, color_ndim),
                ([1, 3], [0, 1]),
            ),
        }
        self.rmtab = {
            "ot_matrix_spin(%d)"
            % (spin_ndim): (lambda: self, None),  # TODO: add proper indices
            "ot_matrix_color(%d)"
            % (color_ndim): (lambda: self, None),  # TODO: add proper indices
        }


def matrix_spin_color(grid, spin_ndim, color_ndim):
    return gpt_object(grid, ot_matrix_spin_color(spin_ndim, color_ndim))


class ot_vector_spin_color(ot_base):
    def __init__(self, spin_ndim, color_ndim):
        self.spin_ndim = spin_ndim
        self.color_ndim = color_ndim
        self.__name__ = "ot_vector_spin_color(%d,%d)" % (spin_ndim, color_ndim)
        self.nfloats = 2 * color_ndim * spin_ndim
        self.shape = (spin_ndim, color_ndim)
        self.v_otype = ["ot_vspin%dcolor%d" % (spin_ndim, color_ndim)]
        self.ot_matrix = "ot_matrix_spin_color(%d,%d)" % (spin_ndim, color_ndim)
        self.spintrace = (0, lambda: ot_vector_color(color_ndim))
        self.colortrace = (1, lambda: ot_vector_spin(spin_ndim))
        self.otab = {
            self.__name__: (
                lambda: ot_matrix_spin_color(spin_ndim, color_ndim),
                [(1, 2)],
            ),
        }
        self.itab = {
            self.__name__: (lambda: ot_singlet, ([0, 1], [0, 1])),
        }
        self.rmtab = {
            "ot_matrix_spin(%d)"
            % (spin_ndim): (lambda: self, None),  # TODO: add proper indices
            "ot_matrix_color(%d)"
            % (color_ndim): (lambda: self, None),  # TODO: add proper indices
        }

    def distribute(self, mat, dst, src, zero_lhs):
        if src.otype.__name__ == self.ot_matrix:
            grid = src.grid
            dst_sc, src_sc = gpt_object(grid, self), gpt_object(grid, self)
            for s in range(self.spin_ndim):
                for c in range(self.color_ndim):
                    gpt.qcd.prop_to_ferm(src_sc, src, s, c)
                    if zero_lhs:
                        dst_sc[:] = 0
                    mat(dst_sc, src_sc)
                    gpt.qcd.ferm_to_prop(dst, dst_sc, s, c)
        else:
            assert 0


def vector_spin_color(grid, spin_ndim, color_ndim):
    return gpt_object(grid, ot_vector_spin_color(spin_ndim, color_ndim))


###
# Basic vectors for coarse grid
class ot_vsinglet4(ot_base):
    nfloats = 2 * 4
    shape = (4,)
    v_otype = ["ot_vsinglet4"]


class ot_vsinglet5(ot_base):
    nfloats = 2 * 5
    shape = (5,)
    v_otype = ["ot_vsinglet5"]


class ot_vsinglet10(ot_base):
    nfloats = 2 * 10
    shape = (10,)
    v_otype = ["ot_vsinglet10"]


class ot_vsinglet(ot_base):
    fundamental = {
        4: ot_vsinglet4,
        5: ot_vsinglet5,
        10: ot_vsinglet10,
    }

    def __init__(self, n):
        self.__name__ = "ot_vsinglet(%d)" % n
        self.nfloats = 2 * n
        self.shape = (n,)
        self.transposed = None
        self.spintrace = None
        self.colortrace = None
        decomposition = decompose(n, ot_vsinglet.fundamental.keys(), 1)
        self.v_n0, self.v_n1 = get_range(decomposition, 1)
        self.v_idx = range(len(self.v_n0))
        self.v_otype = [ot_vsinglet.fundamental[x].__name__ for x in decomposition]
        self.mtab = {
            "ot_singlet": (lambda: self, None),  # TODO: need to add info on contraction
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }


def vsinglet(grid, n):
    return gpt_object(grid, ot_vsinglet(n))


# and matrices
class ot_msinglet4(ot_base):
    nfloats = 2 * 4 * 4
    shape = (4, 4)
    v_otype = ["ot_msinglet4"]


class ot_msinglet5(ot_base):
    nfloats = 2 * 5 * 5
    shape = (5, 5)
    v_otype = ["ot_msinglet5"]


class ot_msinglet10(ot_base):
    nfloats = 2 * 10 * 10
    shape = (10, 10)
    v_otype = ["ot_msinglet10"]


class ot_msinglet(ot_base):
    fundamental = {
        4: ot_msinglet4,
        5: ot_msinglet5,
        10: ot_msinglet10,
    }

    def __init__(self, n):
        self.__name__ = "ot_msinglet(%d)" % n
        self.nfloats = 2 * n * n
        self.shape = (n, n)
        self.transposed = None
        self.spintrace = None
        self.colortrace = None
        self.vector_type = ot_vsinglet(n)
        self.mtab = {
            "ot_singlet": (lambda: self, None),
            "ot_vsinglet(%d)" % n: (lambda: self.vector_type, (1, 0)),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }
        decomposition = decompose(n, ot_msinglet.fundamental.keys(), 2)
        self.v_n0, self.v_n1 = get_range(decomposition, 2)
        self.v_idx = range(len(self.v_n0))
        self.v_otype = [ot_msinglet.fundamental[x].__name__ for x in decomposition]


def msinglet(grid, n):
    return gpt_object(grid, ot_msinglet(n))


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
        root = a
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
            "ot_matrix_su3_fundamental",
            "ot_matrix_su2_fundamental",
            "ot_matrix_su2_adjoint",
            "ot_vsinglet4",
            "ot_vsinglet5",
            "ot_vsinglet10",
            "ot_msinglet4",
            "ot_msinglet5",
            "ot_msinglet10",
        ]
    )

    assert root in known_types
    return eval(root + args)


###
# aliases
def complex(grid):
    return singlet(grid)


def vcomplex(grid, n):
    return vsinglet(grid, n)


def mcomplex(grid, n):
    return msinglet(grid, n)


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
