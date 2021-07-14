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

metropolis = g.algorithms.markov.metropolis(rng)


def measure(x):
    return [g.sum(x).real / grid.fsites, g.norm2(x) / grid.fsites]


eps = 0.08
for i in range(10):
    rng.uniform_element(dx)
    trial = metropolis(x)
    f_before = g.norm2(x)
    x += eps * dx
    f_after = g.norm2(x)
    accept = trial(f_after, f_before)

res = measure(x)

res_ref = [0.0011463214177638292, 0.004613301864105977]
for a, b in zip(res, res_ref):
    eps = abs(a - b) / abs(a)
    g.message(f"Check: {eps}")
    assert eps < 1e-6

# Below checks that things convere to correct value for large N.
# After this, we recompute what we check above.
#
# thermalization
# for step in [0.03, 0.06, 0.08]:
#     tmp = []
#     for i in range(1000):
#         tmp += [metro(step)]
#     tmp = numpy.array(tmp)
#     g.message(f"Acceptance {numpy.mean(tmp[:,0])}")
#
# data = []
# history = []
#
# N = 20000
#
# for i in range(N):
#     for j in range(10):
#         history += [metro(step)]
#     data.append(measure(x))
#
# data = numpy.array(data)
# history = numpy.array(history)
# g.message(f"Acceptance rate = {numpy.mean(history[:,0]):.2f}")
#
# g.message(f"<x>   = {numpy.mean(data[:,0]):6f}, theory = {0.0}")
# g.message(f"<x^2> = {numpy.mean(data[:,1]):6f}, theory = {0.5}")
#
# assert abs(numpy.mean(data[:, 1]) - 0.5) < 1e-4
