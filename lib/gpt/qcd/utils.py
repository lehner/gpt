#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt

def ferm_to_prop(p, f, s, c):
    return cgpt.util_ferm2prop(f.obj,p.obj,s,c,True)

def prop_to_ferm(f, p, s, c):
    return cgpt.util_ferm2prop(f.obj,p.obj,s,c,False)


