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
import cgpt, gpt, numpy
from gpt.default import is_verbose
from gpt.core.expr import factor
from gpt.core.mem import host
from gpt.core.foundation import lattice as foundation, base as foundation_base

mem_book = {}
verbose_lattice_creation = is_verbose("lattice_creation")


def get_mem_book():
    return mem_book


class lattice_view_constructor:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, key):
        pos, tidx, shape = gpt.map_key(self.parent, key)
        return gpt.lattice_view(self.parent, pos, tidx)


def unpack_cache_key(key):
    if isinstance(key, tuple) and isinstance(key[-1], dict):
        cache = key[-1]
        key = key[0:-1]
        if len(key) == 1:
            key = key[0]
        return cache, key

    return None, key


# lattice class
class lattice(factor, foundation_base):
    __array_priority__ = 1000000
    cache = {}
    foundation = foundation

    def __init__(self, first, second=None, third=None):
        self.metadata = {}
        cb = None
        if isinstance(first, gpt.grid):
            self.grid = first
            if isinstance(second, str):
                # from desc
                p = second.split(";")
                self.otype = gpt.str_to_otype(p[0])
                cb = gpt.str_to_cb(p[1])
                self.v_obj = [
                    cgpt.create_lattice(self.grid.obj, t, self.grid.precision.cgpt_dtype)
                    for t in self.otype.v_otype
                ]
            else:
                self.otype = second
                if third is not None:
                    self.v_obj = third
                else:
                    self.v_obj = [
                        cgpt.create_lattice(self.grid.obj, t, self.grid.precision.cgpt_dtype)
                        for t in self.otype.v_otype
                    ]
        elif isinstance(first, gpt.lattice):
            # Note that copy constructor only creates a compatible lattice but does not copy its contents!
            self.grid = first.grid
            self.otype = first.otype
            self.v_obj = [
                cgpt.create_lattice(self.grid.obj, t, self.grid.precision.cgpt_dtype)
                for t in self.otype.v_otype
            ]
            cb = first.checkerboard()
        else:
            raise Exception("Unknown lattice constructor")

        # use first pointer to index page in memory book
        mem_book[self.v_obj[0]] = (
            self.grid,
            self.otype,
            gpt.time(),
            gpt.get_call_stack() if verbose_lattice_creation else None,
        )
        if cb is not None:
            self.checkerboard(cb)

    def __del__(self):
        del mem_book[self.v_obj[0]]
        for o in self.v_obj:
            cgpt.delete_lattice(o)

    def new(self):
        return lattice(self)

    def swap(self, other):
        assert self.grid == other.grid
        assert self.otype == other.otype
        self.v_obj, other.v_obj = other.v_obj, self.v_obj
        self.metadata, other.metadata = other.metadata, self.metadata

    def update(self, v_obj):
        if v_obj != self.v_obj:
            mb = mem_book[self.v_obj[0]]
            del mem_book[self.v_obj[0]]
            self.v_obj = v_obj
            mem_book[self.v_obj[0]] = mb

    def checkerboard(self, val=None):
        if val is None:
            if self.grid.cb.n == 1:
                return gpt.none

            cb = cgpt.lattice_get_checkerboard(self.v_obj[0])  # all have same cb, use 0
            if cb == gpt.even.tag:
                return gpt.even
            elif cb == gpt.odd.tag:
                return gpt.odd
            else:
                assert False
        else:
            if val != gpt.none:
                assert self.grid.cb.n != 1
                for o in self.v_obj:
                    cgpt.lattice_change_checkerboard(o, val.tag)
            return self

    def describe(self):
        # creates a string without spaces that can be used to construct it again (may be combined with self.grid.describe())
        return self.otype.__name__ + ";" + self.checkerboard().__name__

    def nfloats(self):
        return self.otype.nfloats * self.grid.gsites

    def global_bytes(self):
        return self.nfloats() * self.grid.precision.nbytes

    def rank_bytes(self):
        return self.global_bytes() // self.grid.Nprocessors

    @property
    def view(self):
        return lattice_view_constructor(self)

    def __setitem__(self, key, value):
        # unpack cache
        cache, key = unpack_cache_key(key)
        cache_key = None if cache is None else "set"

        # short code path to zero lattice
        if isinstance(key, slice) and key == slice(None, None, None):
            if gpt.util.is_num(value):
                for o in self.v_obj:
                    cgpt.lattice_set_to_number(o, value)
                return

            cache_key = (
                f"{self.otype.__name__}_{self.checkerboard().__name__}_{self.grid.describe()}"
            )
            cache = lattice.cache

        # general code path, map key
        pos, tidx, shape = gpt.map_key(self, key)
        n_pos = len(pos)

        # convert input to proper numpy array
        value = gpt.util.tensor_to_value(value, dtype=self.grid.precision.complex_dtype)
        if value is None:
            value = memoryview(bytearray())

        # needed bytes and optional cyclic upscaling
        nbytes_needed = n_pos * numpy.prod(shape) * self.grid.precision.nbytes * 2
        value = cgpt.copy_cyclic_upscale(value, nbytes_needed)

        # create plan
        if cache_key is None or cache_key not in cache:
            plan = gpt.copy_plan(self, value)
            plan.destination += gpt.lattice_view(self, pos, tidx)
            plan.source += gpt.global_memory_view(
                self.grid,
                [[self.grid.processor, value, 0, value.nbytes]] if value.nbytes > 0 else None,
            )

            # skip optimization if we only use it once
            xp = plan(
                local_only=isinstance(pos, gpt.core.local_coordinates),
                skip_optimize=cache_key is None,
            )
            if cache_key is not None:
                cache[cache_key] = xp
        else:
            xp = cache[cache_key]

        xp(self, value)

    def __getitem__(self, key):
        # unpack cache
        cache, key = unpack_cache_key(key)
        cache_key = None if cache is None else "get"

        # general code path, map key
        pos, tidx, shape = gpt.map_key(self, key)
        n_pos = len(pos)

        # create target
        value = cgpt.ndarray((n_pos, *shape), self.grid.precision.complex_dtype)

        # create plan
        if cache_key is None or cache_key not in cache:
            plan = gpt.copy_plan(value, self)
            plan.destination += gpt.global_memory_view(
                self.grid,
                [[self.grid.processor, value, 0, value.nbytes]] if value.nbytes > 0 else None,
            )
            plan.source += gpt.lattice_view(self, pos, tidx)
            xp = plan()

            if cache_key is not None:
                cache[cache_key] = xp
        else:
            xp = cache[cache_key]

        xp(value, self)

        # if only a single element is returned and we have the full shape,
        # wrap in a tensor
        if len(value) == 1:
            if shape == self.otype.shape:
                return gpt.util.value_to_tensor(value[0], self.otype)
            elif numpy.prod(shape) == 1:
                value = value.item()

        return value

    def mview(self, location=host):
        return [cgpt.lattice_memory_view(self, o, location) for o in self.v_obj]

    def __repr__(self):
        return "lattice(%s,%s)" % (self.otype.__name__, self.grid.precision.__name__)

    def __str__(self):
        if len(self.v_obj) == 1:
            return self.__repr__() + "\n" + cgpt.lattice_to_str(self.v_obj[0])
        else:
            s = self.__repr__() + "\n"
            for i, x in enumerate(self.v_obj):
                s += "-------- %d to %d --------\n" % (
                    self.otype.v_n0[i],
                    self.otype.v_n1[i],
                )
                s += cgpt.lattice_to_str(x)
            return s

    def __iadd__(self, expr):
        gpt.eval(self, expr, ac=True)
        return self

    def __isub__(self, expr):
        gpt.eval(self, -expr, ac=True)
        return self

    def __imatmul__(self, expr):
        gpt.eval(self, expr, ac=False)
        return self

    def __imul__(self, expr):
        gpt.eval(self, self * expr, ac=False)
        return self

    def __itruediv__(self, expr):
        gpt.eval(self, self / expr, ac=False)
        return self

    def __lt__(self, other):
        assert self.otype.data_otype() == gpt.ot_singlet
        assert other.otype.data_otype() == gpt.ot_singlet
        res = gpt.lattice(self)
        params = {"operator": "<"}
        cgpt.binary(res.v_obj[0], self.v_obj[0], other.v_obj[0], params)
        return res

    def __gt__(self, other):
        return other.__lt__(self)
