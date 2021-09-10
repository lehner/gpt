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
from gpt.core.object_type.base import ot_base
import gpt
import cgpt
import numpy

# query cgpt about available sizes
lattice_types = cgpt.lattice_types()
basis_sizes = sorted(
    [int(x[11:]) for x in filter(lambda x: x[0:11] == "ot_msinglet", lattice_types)]
)

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


###
# Singlet
class ot_singlet(ot_base):
    __name__ = "ot_singlet"
    nfloats = 2
    shape = (1,)
    spintrace = (None, None, None)  # do nothing
    colortrace = (None, None, None)
    v_otype = ["ot_singlet"]
    mtab = {
        "ot_singlet": (lambda: ot_singlet, None),
    }

    def data_otype(self=None):
        return ot_singlet

    def identity():
        return 1.0


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
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }

    def identity(self):
        return gpt.matrix_color(numpy.identity(self.shape[0]), self.shape[0])


class ot_vector_color(ot_base):
    def __init__(self, ndim):
        self.__name__ = "ot_vector_color(%d)" % ndim
        self.nfloats = 2 * ndim
        self.shape = (ndim,)
        self.v_otype = ["ot_vcolor%d" % ndim]
        self.spintrace = (None, None, None)
        self.colortrace = (None, None, None)
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
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }

    def identity(self):
        return gpt.matrix_spin(numpy.identity(self.shape[0]), self.shape[0])


class ot_vector_spin(ot_base):
    def __init__(self, ndim):
        self.__name__ = "ot_vector_spin(%d)" % ndim
        self.nfloats = 2 * ndim
        self.shape = (ndim,)
        self.v_otype = ["ot_vspin%d" % ndim]
        self.spintrace = (None, None, None)
        self.colortrace = (None, None, None)
        self.mtab = {
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }
        self.otab = {self.__name__: (lambda: ot_matrix_spin(ndim), [])}
        self.itab = {self.__name__: (lambda: ot_singlet, (0, 0))}


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
            self.__name__: (lambda: self, ([1, 3], [0, 2]), (0, 2, 1, 3)),
            "ot_vector_spin_color(%d,%d)"
            % (spin_ndim, color_ndim): (
                lambda: ot_vector_spin_color(spin_ndim, color_ndim),
                ([1, 3], [0, 1]),
            ),
            "ot_matrix_spin(%d)" % (spin_ndim): (lambda: self, (1, 0), (0, 3, 1, 2)),
            "ot_matrix_color(%d)" % (color_ndim): (lambda: self, (3, 0)),
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_matrix_spin(%d)" % (spin_ndim): (lambda: self, (1, 0)),
            "ot_matrix_color(%d)" % (color_ndim): (lambda: self, (1, 2), (1, 2, 0, 3)),
            "ot_singlet": (lambda: self, None),
        }

    def identity(self):
        return gpt.matrix_spin_color(
            numpy.multiply.outer(
                numpy.identity(self.shape[0]), numpy.identity(self.shape[2])
            ),
            self.shape[0],
            self.shape[2],
        )


class ot_vector_spin_color(ot_base):
    def __init__(self, spin_ndim, color_ndim):
        self.spin_ndim = spin_ndim
        self.color_ndim = color_ndim
        self.__name__ = "ot_vector_spin_color(%d,%d)" % (spin_ndim, color_ndim)
        self.nfloats = 2 * color_ndim * spin_ndim
        self.shape = (spin_ndim, color_ndim)
        self.v_otype = ["ot_vspin%dcolor%d" % (spin_ndim, color_ndim)]
        self.ot_matrix = "ot_matrix_spin_color(%d,%d)" % (spin_ndim, color_ndim)
        self.spintrace = (None, None, None)
        self.colortrace = (None, None, None)
        self.otab = {
            self.__name__: (
                lambda: ot_matrix_spin_color(spin_ndim, color_ndim),
                [(1, 2)],
            ),
        }
        self.itab = {
            self.__name__: (lambda: ot_singlet, ([0, 1], [0, 1])),
        }
        self.mtab = {
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_matrix_spin(%d)"
            % (spin_ndim): (lambda: self, None),  # TODO: add proper indices
            "ot_matrix_color(%d)"
            % (color_ndim): (lambda: self, None),  # TODO: add proper indices
            "ot_singlet": (lambda: self, None),
        }

    def distribute(self, mat, dst, src, zero_lhs):
        src, dst = gpt.util.to_list(src), gpt.util.to_list(dst)
        if src[0].otype.__name__ == self.ot_matrix:
            assert dst[0].otype.__name__ == self.ot_matrix
            src_grid = src[0].grid
            dst_grid = dst[0].grid
            n = self.spin_ndim * self.color_ndim * len(src)
            dst_sc = [gpt.gpt_object(dst_grid, self) for i in range(n)]
            src_sc = [gpt.gpt_object(src_grid, self) for i in range(n)]
            for i in range(len(src)):
                for s in range(self.spin_ndim):
                    for c in range(self.color_ndim):
                        idx = c + self.color_ndim * (s + self.spin_ndim * i)
                        gpt.qcd.prop_to_ferm(src_sc[idx], src[i], s, c)
                        if zero_lhs:
                            dst_sc[idx][:] = 0
            mat(dst_sc, src_sc)
            for i in range(len(src)):
                for s in range(self.spin_ndim):
                    for c in range(self.color_ndim):
                        idx = c + self.color_ndim * (s + self.spin_ndim * i)
                        gpt.qcd.ferm_to_prop(dst[i], dst_sc[idx], s, c)
        else:
            raise TypeError(
                f"Unexpected type {src[0].otype.__name__} <> {self.ot_matrix}"
            )


###
# Basic vectors for coarse grid
class ot_vector_singlet_base(ot_base):
    def __init__(self, n):
        self.nfloats = 2 * n
        self.shape = (n,)
        self.v_otype = [f"ot_vsinglet{n}"]


class ot_vector_singlet(ot_base):
    fundamental = {n: ot_vector_singlet_base(n) for n in basis_sizes}

    def __init__(self, n):
        self.__name__ = "ot_vector_singlet(%d)" % n
        self.nfloats = 2 * n
        self.shape = (n,)
        self.transposed = None
        self.spintrace = (None, None, None)
        self.colortrace = (None, None, None)
        decomposition = decompose(n, ot_vector_singlet.fundamental.keys(), 1)
        self.v_n0, self.v_n1 = get_range(decomposition, 1)
        self.v_idx = range(len(self.v_n0))
        self.v_otype = [
            ot_vector_singlet.fundamental[x].v_otype[0] for x in decomposition
        ]
        self.mtab = {
            "ot_singlet": (lambda: self, None),  # TODO: need to add info on contraction
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }
        self.itab = {
            self.__name__: (lambda: ot_singlet, (0, 0)),
        }


# and matrices
class ot_matrix_singlet_base(ot_base):
    def __init__(self, n):
        self.nfloats = 2 * n * n
        self.shape = (n, n)
        self.v_otype = [f"ot_msinglet{n}"]


class ot_matrix_singlet(ot_base):
    fundamental = {n: ot_matrix_singlet_base(n) for n in basis_sizes}

    def __init__(self, n):
        self.__name__ = "ot_matrix_singlet(%d)" % n
        self.nfloats = 2 * n * n
        self.shape = (n, n)
        self.transposed = (1, 0)
        self.spintrace = (None, None, None)
        self.colortrace = (0, 1, lambda: ot_singlet)
        self.vector_type = ot_vector_singlet(n)
        self.mtab = {
            self.__name__: (lambda: self, (1, 0)),
            "ot_singlet": (lambda: self, None),
            "ot_vector_singlet(%d)" % n: (lambda: self.vector_type, (1, 0)),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }
        decomposition = decompose(n, ot_matrix_singlet.fundamental.keys(), 2)
        self.v_n0, self.v_n1 = get_range(decomposition, 2)
        self.v_idx = range(len(self.v_n0))
        self.v_otype = [
            ot_matrix_singlet.fundamental[x].v_otype[0] for x in decomposition
        ]

    def identity(self):
        return gpt.matrix_singlet(numpy.identity(self.shape[0]), self.shape[0])
