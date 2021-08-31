#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Production code to generate fine-grid basis vectorscoarse-grid eigenvectors using existing
#
import gpt as g

# parameters
fn = g.default.get("--params", "params.txt")
params = g.params(fn, verbose=True)

# load configuration
U = params["config"]

# matrix to use
fmatrix = params["fmatrix"](U)
op = params["op"](fmatrix)
grid = op.grid[0]

# implicitly restarted lanczos
irl = params["method_evec"]

# run
start = g.vspincolor(grid)
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
start.checkerboard(g.odd)  # traditionally, calculate odd-site vectors

try:
    basis, ev = g.load("checkpoint", grids=grid)
except g.LoadError:
    basis, ev = irl(op, start, params["checkpointer"])
    g.save("checkpoint", (basis, ev))
