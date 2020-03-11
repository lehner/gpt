#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
import gpt

mem_book = {
}

def meminfo():
    gpt.message("=============================================")
    gpt.message("            GPT Memory report                ")
    gpt.message("=============================================")
    tot_gb = 0.0
    for page in mem_book:
        grid = mem_book[page][0]
        otype = mem_book[page][1]
        gb = grid.gsites * grid.precision.nbytes * otype.nfloats / 1024.**3.
        tot_gb += gb
        gpt.message(" %X -> grid = %s, prec = %s, otype = %s | %g GB" % 
                    (page,grid.gdimensions,grid.precision,otype,gb))
    gpt.message("=============================================")
    gpt.message("   Total: %g GB " % tot_gb)
    gpt.message("=============================================")


class lattice:
    def __init__(self, first, second = None, third = None):
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
        mem_book[self.obj] = (self.grid,self.otype)

    def __del__(self):
        del mem_book[self.obj]
        cgpt.delete_lattice(self.obj)

    def __setitem__(self, key, value):
        if key == slice(None,None,None):
            key = ()
        
        assert(type(key) == tuple)
        cgpt.lattice_set_val(self.obj, key, value)

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

    def __imatmul__(self, expr):
        gpt.eval(self,expr,ac=False)
        return self
