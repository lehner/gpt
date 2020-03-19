#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt, gpt

def cshift(first, second, third, fourth = None):

    if type(first) == gpt.lattice and type(second) == gpt.lattice and not fourth is None:
        t=first
        l=gpt.eval(second)
        d=third
        o=fourth
    else:
        l=gpt.eval(first)
        d=second
        o=third
        t=gpt.lattice(l)

    cgpt.cshift(t.obj,l.obj,d,o)
    return t

def copy(first, second = None):

    if not second is None:
        t=first
        l=second
    else:
        l=first
        t=gpt.lattice(l)

    cgpt.copy(t.obj,l.obj)
    return t

def convert(first, second):
    if type(first) == gpt.lattice and type(second) == gpt.lattice:
        cgpt.convert(first.obj,second.obj)
        return first
    elif second == gpt.single or second == gpt.double:
        if type(first) == list:
            src_grid=first[0].grid
        else:
            src_grid=first.grid
        dst_prec=second
        dst_grid=gpt.grid(src_grid.gdimensions,dst_prec,src_grid.cb)
        if type(first) == list:
            dst = [ convert(gpt.lattice(dst_grid, src.otype),src) for src in first ]
        else:
            src = first
            dst = convert(gpt.lattice(dst_grid, src.otype),src)
        return dst
    else:
        assert(0)

def norm2(l):
    if type(l) == gpt.tensor:
        return l.norm2()
    l=gpt.eval(l)
    return cgpt.lattice_norm2(l.obj)

def innerProduct(a,b):
    if type(a) == gpt.tensor and type(b) == gpt.tensor:
        return gpt.adj(a) * b
    a=gpt.eval(a)
    b=gpt.eval(b)
    return cgpt.lattice_innerProduct(a.obj,b.obj)

def axpy_norm(d, a, x, y):
    x=gpt.eval(x)
    y=gpt.eval(y)
    return cgpt.lattice_axpy_norm(d.obj,a,x.obj,y.obj)

def slice(x,dim):
    x=gpt.eval(x)
    return [ gpt.util.value_to_tensor(v,x.otype) for v in cgpt.lattice_slice(x.obj,dim) ]
