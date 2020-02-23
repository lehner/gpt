#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt, gpt

def cshift(first, second, third, fourth = None):

    if type(first) == gpt.lattice and type(second) == gpt.lattice and not fourth is None:
        t=first
        l=second
        d=third
        o=fourth
    else:
        l=first
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

def mul(first, second, third = None):

    if not third is None:
        t=first
        a=second
        b=third
    else:
        a=first
        b=second

        t=gpt.lattice(a.grid,gpt.otype.mul(a.otype,b.otype))

    cgpt.lattice_mul(t.obj,a.obj,b.obj)
    return t
