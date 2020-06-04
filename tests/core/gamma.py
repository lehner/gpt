#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g

l=g.vspincolor(g.grid([4,4,4,4],g.double))

g.random("test").cnormal(l)

dst=g.lattice(l)
ref=g.lattice(l)

dims=["X","Y","Z","T"]

assert(g.norm2(l) > 1e-7)

for i,d1 in enumerate(dims):
    for j,d2 in enumerate(dims):
        if i<j:
            dst @= g.gamma[d1] * g.gamma[d2] * l
            dst -= g.gamma[d2] * g.gamma[d1] * l
            dst *= 1 / 2.
            ref @= g.gamma["Sigma%s%s" % (d1,d2) ]*l
            eps = g.norm2(dst - ref) / g.norm2(l)
            g.message("Test Sigma%s%s: " % (d1,d2),eps)
            assert(eps == 0.0)

dst @= g.gamma["X"] * g.gamma["Y"] * g.gamma["Z"] * g.gamma["T"] * l
ref @= g.gamma[5] * l
eps = g.norm2(dst - ref) / g.norm2(l)
g.message("Test Gamma5: ",eps)
assert(eps == 0.0)

# test creating spinor projections as expession templates
P=g.gamma[0]*g.gamma[1] - g.gamma[2]*g.gamma[3]

dst @= P * l
ref @= g.gamma[0]*g.gamma[1]*l - g.gamma[2]*g.gamma[3]*l
eps = g.norm2(dst - ref) / g.norm2(l)
g.message("Test Regular Expression: ",eps)
assert(eps == 0.0)



g.message("All tests passed")



