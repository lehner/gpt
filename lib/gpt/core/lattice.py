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
import cgpt
import gpt
import numpy
from time import time

mem_book = {
}

def meminfo():
    fmt=" %-8s %-30s %-12s %-15s %-12s %-16s %-20s"
    gpt.message("==========================================================================================")
    gpt.message("                                 GPT Memory Report                ")
    gpt.message("==========================================================================================")
    gpt.message(fmt % ("Index","Grid","Precision","OType", "CBType", "Size/GB", "Created at time"))
    tot_gb = 0.0
    for i,page in enumerate(mem_book):
        grid,otype,created = mem_book[page]
        gb = grid.gsites * grid.precision.nbytes * otype.nfloats / grid.cb.n / 1024.**3.
        tot_gb += gb
        gpt.message(fmt % (i,grid.gdimensions,grid.precision.__name__,
                           otype.__name__,grid.cb.__name__,gb,"%.6f s" % created))
    gpt.message("==========================================================================================")
    gpt.message("   Total: %g GB " % tot_gb)
    gpt.message("==========================================================================================")


class lattice:
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
                self.obj = cgpt.create_lattice(self.grid.obj, self.otype, self.grid.precision)
            else:
                self.otype = second
                if not third is None:
                    self.obj = third
                else:
                    self.obj = cgpt.create_lattice(self.grid.obj, self.otype, self.grid.precision)
        elif type(first) == gpt.lattice:
            # Note that copy constructor only creates a compatible lattice but does not copy its contents!
            self.grid = first.grid
            self.otype = first.otype
            self.obj = cgpt.create_lattice(self.grid.obj, self.otype, self.grid.precision)
        else:
            raise Exception("Unknown lattice constructor")
        mem_book[self.obj] = (self.grid,self.otype,gpt.time())
        if not cb is None:
            self.checkerboard(cb)

    def __del__(self):
        del mem_book[self.obj]
        cgpt.delete_lattice(self.obj)

    def checkerboard(self, val = None):
        if val is None:
            if self.grid.cb != gpt.redblack:
                return gpt.none

            cb=cgpt.get_checkerboard(self.obj)
            if cb == gpt.even.tag:
                return gpt.even
            elif cb == gpt.odd.tag:
                return gpt.odd
            else:
                assert(0)
        else:
            if val != gpt.none:
                assert(self.grid.cb == gpt.redblack)
                cgpt.lattice_change_checkerboard(self.obj,val.tag)

    def describe(self):
        # creates a string without spaces that can be used to construct it again (may be combined with self.grid.describe())
        return self.otype.__name__ + ";" + self.checkerboard().__name__

    def __setitem__(self, key, value):
        if type(key) == slice:
            if key == slice(None,None,None):
                key = ()

        if type(key) == tuple:
            cgpt.lattice_set_val(self.obj, key, gpt.util.tensor_to_value(value))
        elif type(key) == numpy.ndarray:
            cgpt.lattice_import(self.obj, key, value)
        else:
            assert(0)

    def __getitem__(self, key):
        if type(key) == tuple:
            return gpt.util.value_to_tensor(cgpt.lattice_get_val(self.obj, key), self.otype)
        elif type(key) == numpy.ndarray:
            return cgpt.lattice_export(self.obj,key)
        else:
            assert(0)

    def mview(self):
        return cgpt.lattice_memory_view(self.obj)

    def __repr__(self):
        return "lattice(%s,%s)" % (self.otype,self.grid.precision)

    def __str__(self):
        return cgpt.lattice_to_str(self.obj)

    def __rmul__(self, l):
        return gpt.expr(l) * gpt.expr(self)

    def __mul__(self, l):
        return gpt.expr(self) * gpt.expr(l)

    def __truediv__(self, l):
        assert(gpt.util.isnum(l))
        return gpt.expr(self) * (1.0/l)

    def __add__(self, l):
        return gpt.expr(self) + gpt.expr(l)

    def __sub__(self, l):
        return gpt.expr(self) - gpt.expr(l)

    def __neg__(self):
        return gpt.expr(self) * (-1.0)

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
