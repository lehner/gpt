#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import sys, os
import numpy as np

# workdir
if "WORK_DIR" in os.environ:
    work_dir = os.environ["WORK_DIR"]
else:
    work_dir = "."

# grids
fgrid = g.grid([12,8,8,8,16],g.single,g.redblack)
cgrid = g.grid([1,4,4,4,4],g.single)

# vectors
nbasis = 20
nevec = 30
rng = g.random("test")
basis = [ g.vspincolor(fgrid) for i in range(nbasis) ]
cevec = [ g.vcomplex(cgrid,nbasis) for i in range(nevec) ]
feval = [ rng.normal(mu = 2.0,sigma = 0.5).real for i in range(nevec) ]
for b in basis:
    b.checkerboard(g.odd)
rng.cnormal([basis,cevec])
for i in range(2):
    g.block.orthonormalize(cgrid, basis)

# save in fixed layout
g.save(
    f"{work_dir}/cevec",
    [basis, cevec, feval],
    g.format.cevec(
        {"nsingle": len(basis) // 2, "max_read_blocks": 16, "mpi": [1, 2, 2, 2, 2]}
    ),
)

# and load again to verify
basis2, cevec2, feval2 = g.load(
    f"{work_dir}/cevec", {"grids": fgrid}
)

assert len(basis) == len(basis2)
assert len(cevec) == len(cevec2)
assert len(feval) == len(feval2)

pos = g.coordinates(basis[0])
eps = 0.0
for i in range(len(basis)):
    A = basis[i][pos]
    B = basis2[i][pos]
    eps += fgrid.globalsum(float(np.linalg.norm(A - B)))
g.message("Test basis: %g" % (eps))

pos = g.coordinates(cevec[0])
eps = 0.0
for i in range(len(cevec)):
    A = cevec[i][pos]
    B = cevec2[i][pos]
    eps += fgrid.globalsum(float(np.linalg.norm(A - B)))
g.message("Test cevec: %g" % (eps))

eps = 0.0
for i in range(len(feval)):
    eps += (feval[i] - feval2[i]) ** 2.0
g.message("Test eval: %g" % (eps))
