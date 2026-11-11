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
import cgpt, gpt
import numpy as np
import gc, sys, types


# Per-grid state (constant fields, kernels, workspaces) of the whole process.
# The key is (grid.cache_key(),) + tag, so identical python grids share
# entries (see grid.cache_key).  Grids never refer to this cache (a
# grid holding lattices on itself is a reference cycle that the lattice
# memory book keeps alive), and cgpt never frees a grid either: entries live
# until grid_cache_clear().
_grid_cache = {}


def grid_cache(grid, tag, create):
    # the entry for (grid.cache_key(),) + tag, created by create() on first use
    key = (grid.cache_key(),) + tuple(tag)
    x = _grid_cache.get(key)
    if x is None:
        x = create()
        _grid_cache[key] = x
    return x


def grid_cache_clear():
    _grid_cache.clear()


def _cache_referents(x):
    # the references followed from inside the cache: containers, instance
    # attributes and closures; never types, modules, code, grids, and not into
    # lattices (the leaves)
    if isinstance(x, gpt.lattice) or isinstance(x, grid):
        return []
    if isinstance(x, (type, types.ModuleType, types.CodeType, types.BuiltinFunctionType)):
        return []
    if isinstance(x, types.FunctionType):
        return [c.cell_contents for c in (x.__closure__ or ()) if _cell_filled(c)]
    if isinstance(x, types.MethodType):
        return [x.__self__, x.__func__]
    return [r for r in gc.get_referents(x) if not isinstance(r, type)]


def _cell_filled(c):
    try:
        c.cell_contents
        return True
    except ValueError:
        return False


def grid_cache_bytes():
    # (held, freeable): bytes of the lattice fields on all ranks reachable from
    # g.grid_cache, and the part that g.grid_cache_clear() would release, i.e.
    # not also reachable from a reference outside the cache.  An object is
    # referenced from outside if its reference count exceeds the references
    # found inside; references the walk does not see count as outside, so
    # freeable is a lower bound.
    objs = {}
    edges = {}
    internal = {}
    stack = []

    def visit(refs):
        # count the references; walk the objects that can hold others
        for r in refs:
            internal[id(r)] = internal.get(id(r), 0) + 1
            if id(r) not in objs and gc.is_tracked(r):
                objs[id(r)] = r
                stack.append(id(r))

    visit(list(_grid_cache.values()))
    while stack:
        k = stack.pop()
        refs = _cache_referents(objs[k])
        edges[k] = [id(r) for r in refs]
        visit(refs)
        del refs
    # (no other local refers to them here: 2 = the argument of getrefcount
    # and objs)
    held = [k for k in objs if sys.getrefcount(objs[k]) - 2 > internal[k]]
    reached = set(held)
    while held:
        k = held.pop()
        for r in edges.get(k, []):
            if r in objs and r not in reached:
                reached.add(r)
                held.append(r)
    total, freeable = 0, 0
    for k, x in objs.items():
        if isinstance(x, gpt.lattice):
            total += x.global_bytes()
            if k not in reached:
                freeable += x.global_bytes()
    return total, freeable


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
    assert len(p) % 3 == 0
    parent = None
    for i in reversed(range(len(p) // 3)):
        fdimensions = [int(x) for x in p[3 * i + 0].strip("[]").split(",")]
        precision = gpt.str_to_precision(p[3 * i + 1])
        cb = str_to_checkerboarding(p[3 * i + 2], len(fdimensions))
        obj = None
        mpi = None if parent is None else parent.mpi
        parent = grid(fdimensions, precision, cb, obj, parent=parent, mpi=mpi)
    return parent


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
    def __init__(self, fdimensions, precision, cb=None, obj=None, mpi=None, parent=None):
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
                fdimensions, precision.cgpt_dtype, cb.cb_mask, cb.simd_mask, self.mpi, parent_obj
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

        # make sure Grid does not change conventions
        assert self.mpi_strides() @ self.processor_coor == self.processor

    def cache_key(self):
        # the key of caches of per-grid state: cgpt hands identical grids
        # (dimensions, simd layout, checkerboard mask, mpi layout, parent) one
        # shared Grid object (same obj), which has no precision; a gpt grid is
        # that layout plus a precision
        return (self.obj, self.precision)

    def describe(
        self,
    ):  # creates a string without spaces that can be used to construct it again, this should only describe the grid geometry not the mpi
        s = (
            str(self.fdimensions) + ";" + self.precision.__name__ + ";" + self.cb.__name__
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

    def local_indices(self, *dimensions):
        assert len(dimensions) == 1
        # multi dimensions later
        return (
            np.arange(self.ldimensions[dimensions[0]])
            + self.processor_coor[dimensions[0]] * self.ldimensions[dimensions[0]]
        )

    def simd(self):
        return cgpt.grid_get_simd(self.obj)

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
            self.cb.simd_mask[0:dimension] + [simd_mask] + self.cb.simd_mask[dimension:],
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

    def mpi_strides(self):
        # Grid's mpi strides such that mpi_strides @ processor_coor == processor
        return np.flip(np.concatenate(([1], np.cumprod(np.flip(self.mpi))[:-1])))

    def cartesian_rank(self):
        # cartesian rank with x running fastests
        # this is used for IO file index mapping
        # this does not agree with Grid's mapping
        rank = 0
        for i in reversed(range(self.nd)):
            rank = rank * self.mpi[i] + self.processor_coor[i]
        # rank = coor[3] * size[2] * size[1] * size[0] + coor[2] * size[1] * size[0] + coor[1] * size[0] + coor[0]
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
        return self.precision.global_sum_policy(self, x)

    def broadcast(self, root, array):
        cgpt.grid_broadcast(self.obj, root, array)

    def exchange(self, send_to, recv_from, send_array, recv_array):
        cgpt.grid_exchange(self.obj, send_to, recv_from, send_array, recv_array)

    def lexicographic_index(self, coordinates):
        # order of coordinates x,y,z,t
        # s = x + Lx*y + Lx*Ly*z + Lx*Ly*Lz*t
        l = [1] + list(np.cumprod(self.fdimensions[:-1]))
        n, d = coordinates.shape
        assert d == self.nd
        return coordinates @ np.array(l)

    def reduce_binary_tree(self, array, functor):
        me = np.copy(array)
        other = np.copy(array)

        planes = int(np.ceil(np.log2(self.Nprocessors)))

        for i in range(planes):
            j = self.processor ^ (2**i)
            if j < self.Nprocessors:
                self.exchange(j, j, me, other)
                functor(me, other)

        self.broadcast(0, me)

        return me

    def reduce_dimensional_reduction(self, array, functor):
        accumulator = np.copy(array)
        for d in range(self.nd):
            gather = np.ndarray(shape=[self.mpi[d]] + list(array.shape), dtype=array.dtype)
            cgpt.grid_gather_dimension(self.obj, accumulator, gather, d)
            for i in range(1, self.mpi[d]):
                functor(accumulator, gather[i])
        self.broadcast(0, accumulator)

        return accumulator

    def reduce(self, array, functor):
        return self.reduce_binary_tree(array, functor)
