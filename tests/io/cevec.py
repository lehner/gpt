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
fgrid = g.grid([4, 8, 8, 8, 16], g.single, g.redblack)
cgrid = g.grid([1, 4, 4, 4, 4], g.single)
# fgrid = g.grid([12, 96, 48, 24, 24], g.single, g.redblack)
# cgrid = g.grid([1, 96//4, 48//4, 24//4, 24//4], g.single)

# vectors
nbasis = 40
nsingle = 10
nevec = 48
rng = g.random("test")
basis = [g.vspincolor(fgrid) for i in range(nbasis)]
cevec = [g.vcomplex(cgrid, nbasis) for i in range(nevec)]
feval = [rng.normal(mu=2.0, sigma=0.5).real for i in range(nevec)]
for b in basis:
    b.checkerboard(g.odd)
rng.cnormal([basis, cevec])
b = g.block.map(cgrid, basis)
for i in range(2):
    b.orthonormalize()

for mpi_layout in [[1, 1, 1, 1, 1], [1, 2, 2, 2, 2]]:

    # save in fixed layout
    g.save(
        f"{work_dir}/cevec",
        [basis, cevec, feval],
        g.format.cevec({"nsingle": nsingle, "max_read_blocks": 8, "mpi": mpi_layout}),
    )

    # and load again to verify
    basis2, cevec2, feval2 = g.load(f"{work_dir}/cevec", {"grids": fgrid})

    assert len(basis) == len(basis2)
    assert len(cevec) == len(cevec2)
    assert len(feval) == len(feval2)

    for i in range(len(basis)):
        eps2 = g.norm2(basis[i] - basis2[i]) / g.norm2(basis[i])
        g.message(f"basis {i} resid {eps2}")
        if i < nsingle:
            assert eps2 == 0.0
        else:
            assert eps2 < 1e-9

    for i in range(len(cevec)):
        eps2 = g.norm2(cevec[i] - cevec2[i]) / g.norm2(cevec[i])
        g.message(f"cevec {i} resid {eps2}")
        assert eps2 < 1e-9

    for i in range(len(feval)):
        assert (feval[i] - feval2[i]) ** 2.0 < 1e-25

    # and load truncated and verify
    for ntrunc in [46, 32, 8]:
        g.message(
            f"""

    Test with ntrunc = {ntrunc}

"""
        )
        basis2, cevec2, feval2 = g.load(
            f"{work_dir}/cevec", {"grids": fgrid, "nmax": ntrunc}
        )

        assert min([len(basis), ntrunc]) == len(basis2)
        assert min([len(cevec), ntrunc]) == len(cevec2)
        assert len(feval) == len(feval2)

        for i in range(len(basis2)):
            eps2 = g.norm2(basis[i] - basis2[i]) / g.norm2(basis[i])
            g.message(f"basis {i} resid {eps2}")
            if i < nsingle:
                assert eps2 == 0.0
            else:
                assert eps2 < 1e-9

        for i in range(len(cevec2)):
            eps2 = g.norm2(cevec[i] - cevec2[i]) / g.norm2(cevec[i])
            g.message(f"cevec {i} resid {eps2}")
            assert eps2 < 1e-9

        for i in range(len(feval2)):
            assert (feval[i] - feval2[i]) ** 2.0 < 1e-25
