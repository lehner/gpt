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
from gpt.core.expr import factor

mem_book = {
}

def get_mem_book():
    return mem_book


class lattice(factor):
    __array_priority__=1000000
    def __init__(self, first, second = None, third = None):
        self.metadata={}
        cb=None
        if type(first) == gpt.grid:
            self.grid = first
            if type(second) == str:
                # from desc
                p=second.split(";")
                self.otype=gpt.str_to_otype(p[0])
                cb=gpt.str_to_cb(p[1])
                self.v_obj = [ cgpt.create_lattice(self.grid.obj, t, self.grid.precision) for t in self.otype.v_otype ]
            else:
                self.otype = second
                if not third is None:
                    self.v_obj = third
                else:
                    self.v_obj = [ cgpt.create_lattice(self.grid.obj, t, self.grid.precision) for t in self.otype.v_otype ]
        elif type(first) == gpt.lattice:
            # Note that copy constructor only creates a compatible lattice but does not copy its contents!
            self.grid = first.grid
            self.otype = first.otype
            self.v_obj = [ cgpt.create_lattice(self.grid.obj, t, self.grid.precision) for t in self.otype.v_otype ]
            cb = first.checkerboard()
        else:
            raise Exception("Unknown lattice constructor")

        # use first pointer to index page in memory book
        mem_book[self.v_obj[0]] = (self.grid,self.otype,gpt.time())
        if not cb is None:
            self.checkerboard(cb)

    def __del__(self):
        del mem_book[self.v_obj[0]]
        for o in self.v_obj:
            cgpt.delete_lattice(o)

    def advise(self, t):
        if type(t) != str:
            t=t.tag
        for o in self.v_obj:
            cgpt.lattice_advise(o,t)
        return self

    def prefetch(self, t):
        if type(t) != str:
            t=t.tag
        for o in self.v_obj:
            cgpt.lattice_prefetch(o,t)
        return self

    def checkerboard(self, val = None):
        if val is None:
            if self.grid.cb.n == 1:
                return gpt.none

            cb=cgpt.lattice_get_checkerboard(self.v_obj[0]) # all have same cb, use 0
            if cb == gpt.even.tag:
                return gpt.even
            elif cb == gpt.odd.tag:
                return gpt.odd
            else:
                assert(0)
        else:
            if val != gpt.none:
                assert(self.grid.cb.n != 1)
                for o in self.v_obj:
                    cgpt.lattice_change_checkerboard(o,val.tag)

    def describe(self):
        # creates a string without spaces that can be used to construct it again (may be combined with self.grid.describe())
        return self.otype.__name__ + ";" + self.checkerboard().__name__

    def map_key(self, key):
        # slices without specified start/stop corresponds to memory view limitation for this rank
        if type(key) == slice and key == slice(None,None,None):
            return ()
        if type(key) == tuple and not all([ type(k) == int for k in key ]):
            key = tuple([ k if type(k) == slice else slice(k,k+1) for k in key ])
            grid=self.grid
            assert(all([ k.step is None for k in key ]))
            assert(len(key) == grid.nd)
            top=[ grid.fdimensions[i] // grid.mpi[i] * grid.processor_coor[i] if k.start is None else k.start for i,k in enumerate(key) ]
            bottom=[ grid.fdimensions[i] // grid.mpi[i] * (1+grid.processor_coor[i]) if k.stop is None else k.stop for i,k in enumerate(key) ]
            key=cgpt.coordinates_from_cartesian_view(top,bottom,self.grid.cb.cb_mask,self.checkerboard().tag,"grid")
        return key

    def __setitem__(self, key, value):
        key = self.map_key(key)
        if type(key) == tuple:
            if len(self.v_obj) == 1:
                cgpt.lattice_set_val(self.v_obj[0], key, gpt.util.tensor_to_value(value))
            elif type(value) == int and value == 0:
                for i in self.otype.v_idx:
                    cgpt.lattice_set_val(self.v_obj[i], key, 0)
            else:
                for i in self.otype.v_idx:
                    cgpt.lattice_set_val(self.v_obj[i], key, gpt.tensor(value.array[self.otype.v_n0[i]:self.otype.v_n1[i]],self.otype.v_otype[i]).array)
        elif type(key) == numpy.ndarray:
            cgpt.lattice_import(self.v_obj, key, value)
        else:
            assert(0)

    def __getitem__(self, key):
        key = self.map_key(key)
        if type(key) == tuple:
            if len(self.v_obj) == 1:
                return gpt.util.value_to_tensor(cgpt.lattice_get_val(self.v_obj[0], key), self.otype)
            else:
                val=cgpt.lattice_get_val(self.v_obj[0], key)
                for i in self.otype.v_idx[1:]:
                    val=numpy.append(val,cgpt.lattice_get_val(self.v_obj[i], key))
                return gpt.util.value_to_tensor(val, self.otype)
        elif type(key) == numpy.ndarray:
            return cgpt.lattice_export(self.v_obj,key)
        else:
            assert(0)

    def mview(self):
        return [ cgpt.lattice_memory_view(o) for o in self.v_obj ]

    def mview_coordinates(self):
        # coordinates are identical for all x \in v_obj
        return cgpt.lattice_memory_view_coordinates(self.v_obj[0])

    def __repr__(self):
        return "lattice(%s,%s)" % (self.otype.__name__,self.grid.precision.__name__)

    def __str__(self):
        if len(self.v_obj) == 1:
            return self.__repr__() + "\n" + cgpt.lattice_to_str(self.v_obj[0])
        else:
            s=self.__repr__() + "\n"
            for i,x in enumerate(self.v_obj):
                s+="-------- %d to %d --------\n" % (self.otype.v_n0[i],self.otype.v_n1[i])
                s+=cgpt.lattice_to_str(x)
            return s

    def __iadd__(self, expr):
        gpt.eval(self,expr,ac=True)
        return self

    def __isub__(self, expr):
        gpt.eval(self,-expr,ac=True)
        return self

    def __imatmul__(self, expr):
        gpt.eval(self,expr,ac=False)
        return self

    def __imul__(self, expr):
        gpt.eval(self,self * expr,ac=False)
        return self

    def __imul__(self, expr):
        gpt.eval(self,self * expr,ac=False)
        return self

    def __itruediv__(self, expr):
        gpt.eval(self,self / expr,ac=False)
        return self
