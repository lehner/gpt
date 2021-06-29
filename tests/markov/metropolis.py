#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno 2021
#
import gpt as g
import numpy
import sys

# grid
grid = g.grid([4, 4, 4, 8], g.single)

rng = g.random("3.14")

# we generate data with probability \prod_i dx_i exp(-\sum x_i^2)
x = g.real(grid)
x[:] = 0
dx = g.lattice(x)

def w(eps, x, dx):
    rng.element(dx, min=-1., max=+1.)
    x @= x + eps*dx

h = lambda x: g.norm2(x)

metro = g.algorithms.markov.metropolis(rng, lambda e: w(e,x,dx), h, x)

# thermalization
for step in [0.03, 0.06, 0.08]:
    tmp = []
    for i in range(1000):
        tmp += [metro(step)]
    tmp = numpy.array(tmp)
    g.message(f'Acceptance {numpy.mean(tmp[:,0])}')

data = []
history = []

N = 20000

def measure(x):
    return [g.sum(x).real / grid.fsites, g.norm2(x) / grid.fsites]

for i in range(N):
    for j in range(10):
        history += [metro(step)]
    data.append( measure(x) )

data = numpy.array(data)
history = numpy.array(history)
g.message(f'Acceptance rate = {numpy.mean(history[:,0]):.2f}')

g.message(f'<x>   = {numpy.mean(data[:,0]):6f}, theory = {0.0}')
g.message(f'<x^2> = {numpy.mean(data[:,1]):6f}, theory = {0.5}')

assert(abs(numpy.mean(data[:,1]) - 0.5) < 1e-4)