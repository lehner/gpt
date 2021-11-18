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
import cgpt, gpt, sys
import numpy as np


class full:
    def __init__(self, nd):
        self.n = 1
        self.cb_mask = [0] * nd
        self.simd_mask = [1] * nd
        self.__name__ = "full"


class redblack:
    def __init__(self, nd):
        self.n = 2
        rbd = min([nd, 4])
        self.cb_mask = [0] * (nd - rbd) + [1] * rbd
        self.simd_mask = [1] * nd
        self.__name__ = "redblack"


class general:
    def __init__(self, n, cb_mask, simd_mask):
        self.n = n
        self.cb_mask = cb_mask
        self.simd_mask = simd_mask
        self.__name__ = "general_%d_%s_%s" % (n, cb_mask, simd_mask)


def str_to_checkerboarding(s, nd):
    if s == "full":
        return full(nd)
    elif s == "redblack":
        return redblack(nd)
    else:
        a = s.split("_")
        if a[0] == "general":
            n = int(a[1])
            cb_mask = [int(x) for x in a[2].strip("[]").split(",")]
            simd_mask = [int(x) for x in a[3].strip("[]").split(",")]
            return general(n, cb_mask, simd_mask)
        assert 0


def grid_from_description(description):
    p = description.split(";")
    fdimensions = [int(x) for x in p[0].strip("[]").split(",")]
    precision = gpt.str_to_precision(p[1])
    cb = str_to_checkerboarding(p[2], len(fdimensions))
    obj = None
    return grid(fdimensions, precision, cb, obj)


def grid_get_mpi_default(fdimensions, cb):
    nd = len(fdimensions)
    tag = "--mpi"

    # first try to find mpi layout for dimension nd
    mpi = gpt.default.get_ivec(tag, None, nd)
    if mpi is None and nd > 4:

        # if not found but dimension is larger than four, we try to extend the four-dimensional grid
        mpi = gpt.default.get_ivec(tag, None, 4)
        if mpi is not None:
            mpi = [1] * (nd - 4) + mpi

    if mpi is None:
        # try trivial layout
        mpi = [1] * nd

    assert nd == len(mpi)
    return mpi


class grid:
    def __init__(
        self, fdimensions, precision, cb=None, obj=None, mpi=None, parent=None
    ):

        self.fdimensions = fdimensions
        self.fsites = np.prod(self.fdimensions)
        self.precision = precision
        self.nd = len(self.fdimensions)

        if cb is None:
            cb = full

        if isinstance(cb, type):
            cb = cb(self.nd)

        self.cb = cb
        if mpi is None:
            # if we live on a split grid, cannot use default mpi layout
            assert parent is None
            self.mpi = grid_get_mpi_default(self.fdimensions, self.cb)
        else:
            self.mpi = mpi

        self.parent = parent
        if parent is None:
            parent_obj = 0
        else:
            parent_obj = parent.obj

        if obj is None:
            self.obj = cgpt.create_grid(
                fdimensions, precision, cb.cb_mask, cb.simd_mask, self.mpi, parent_obj
            )
        else:
            self.obj = obj

        # processor is mpi rank, may not be lexicographical (cartesian) rank
        (
            self.processor,
            self.Nprocessors,
            self.processor_coor,
            self.gdimensions,
            self.ldimensions,
            self.srank,
            self.sranks,
        ) = cgpt.grid_get_processor(self.obj)
        self.gsites = np.prod(self.gdimensions)

    def describe(
        self,
    ):  # creates a string without spaces that can be used to construct it again, this should only describe the grid geometry not the mpi
        s = (
            str(self.fdimensions)
            + ";"
            + self.precision.__name__
            + ";"
            + self.cb.__name__
        ).replace(" ", "")
        if self.parent is not None:
            s += ";" + self.parent.describe()
        return s

    def converted(self, dst_precision):
        if dst_precision == self.precision:
            return self
        if self.parent is None:
            parent = None
        else:
            parent = self.parent.converted(dst_precision)
        return grid(
            self.fdimensions,
            dst_precision,
            cb=self.cb,
            obj=None,
            mpi=self.mpi,
            parent=parent,
        )

    def checkerboarded(self, cb):
        if cb == self.cb:
            return self
        if self.parent is None:
            parent = None
        else:
            parent = self.parent.checkerboarded(cb)
        return grid(
            self.fdimensions,
            self.precision,
            cb=cb,
            obj=None,
            mpi=self.mpi,
            parent=parent,
        )

    def split(self, mpi_split, fdimensions):
        return grid(fdimensions, self.precision, self.cb, None, mpi_split, self)

    def inserted_dimension(self, dimension, extent, cb_mask=None, simd_mask=1):
        if cb_mask is None and self.cb.n == 1:
            cb_mask = 0
        assert cb_mask is not None
        cb = general(
            self.cb.n,
            self.cb.cb_mask[0:dimension] + [cb_mask] + self.cb.cb_mask[dimension:],
            self.cb.simd_mask[0:dimension]
            + [simd_mask]
            + self.cb.simd_mask[dimension:],
        )

        if self.parent is None:
            parent = None
            mpi = None
        else:
            parent = self.parent.inserted_dimension(dimension, extent)
            mpi = self.mpi[0:dimension] + [1] + self.mpi[dimension:]

        return grid(
            self.fdimensions[0:dimension] + [extent] + self.fdimensions[dimension:],
            self.precision,
            cb=cb,
            obj=None,
            mpi=mpi,
            parent=parent,
        )

    def removed_dimension(self, dimension):
        assert 0 <= dimension and dimension < self.nd
        cb = general(
            self.cb.n,
            self.cb.cb_mask[0:dimension] + self.cb.cb_mask[dimension + 1 :],
            self.cb.simd_mask[0:dimension] + self.cb.simd_mask[dimension + 1 :],
        )

        if self.parent is None:
            parent = None
        else:
            parent = self.parent.removed_dimension(dimension)

        return grid(
            self.fdimensions[0:dimension] + self.fdimensions[dimension + 1 :],
            self.precision,
            cb=cb,
            obj=None,
            mpi=None,
            parent=parent,
        )

    def cartesian_rank(self):
        rank = 0
        for i in reversed(range(self.nd)):
            rank = rank * self.mpi[i] + self.processor_coor[i]
        return rank

    def __str__(self):
        s = f"fdimensions = {self.fdimensions}; mpi = {self.mpi}; precision = {self.precision.__name__}; checkerboard = {self.cb.__name__}"
        if self.parent is not None:
            s += " split from " + self.parent.__str__()
        return s

    def __del__(self):
        cgpt.delete_grid(self.obj)

    def barrier(self):
        cgpt.grid_barrier(self.obj)

    def globalsum(self, x):
        if type(x) == gpt.tensor:
            cgpt.grid_globalsum(self.obj, x.array)
            return x
        else:
            return cgpt.grid_globalsum(self.obj, x)
