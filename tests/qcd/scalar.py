#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno 2021
#
import gpt as g
import numpy
import sys

# grid
grid = g.grid([8, 8, 8, 8], g.double)

rng = g.random("scalar!")

phi = g.real(grid)
rng.element(phi)

actions = [g.qcd.scalar.action.mass_term(), g.qcd.scalar.action.phi4(0.119, 0.01)]

for a in actions:
    g.message(a.__name__)
    a.assert_gradient_error(rng, phi, phi, 1e-5, 1e-7)
