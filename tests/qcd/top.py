import gpt as g
import matplotlib.pyplot as plt
import numpy as np

NS=4
NT=4

# load configuration
rng = g.random("test")
grid = g.grid([NS,NS,NS,NT], g.double)
U = g.qcd.gauge.random(grid, rng)

Q=g.qcd.gauge.topological_charge(U)
eps=abs(Q-0.03244485592394074)
assert eps < 1e-13
Q=g.qcd.gauge.topological_charge_5LI(U)
eps=abs(Q-0.05826779405836782)
assert eps < 1e-13
