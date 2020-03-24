#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
import gpt
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
        if type(first) == gpt.grid and not second is None and not third is None:
            grid = first
            otype = second
            obj = third
            self.grid = grid
            self.otype = otype
            self.obj = obj
        elif type(first) == gpt.grid and not second is None:
            grid = first
            otype = second
            self.grid = grid
            self.otype = otype
            self.obj = cgpt.create_lattice(self.grid.obj, self.otype, self.grid.precision)
        elif type(first) == gpt.lattice:
            # Note that copy constructor only creates a compatible lattice but does not copy its contents!
            self.grid = first.grid
            self.otype = first.otype
            self.obj = cgpt.create_lattice(self.grid.obj, self.otype, self.grid.precision)
        else:
            raise Exception("Unknown lattice constructor")
        mem_book[self.obj] = (self.grid,self.otype,gpt.time())

    def __del__(self):
        del mem_book[self.obj]
        cgpt.delete_lattice(self.obj)

    def __setitem__(self, key, value):
        if key == slice(None,None,None):
            key = ()
        
        assert(type(key) == tuple)
        cgpt.lattice_set_val(self.obj, key, gpt.util.tensor_to_value(value))

    def __getitem__(self, key):
        assert(type(key) == tuple)
        return gpt.util.value_to_tensor(cgpt.lattice_get_val(self.obj, key), self.otype)

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
