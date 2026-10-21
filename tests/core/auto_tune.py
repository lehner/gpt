#!/usr/bin/env python3
#
# Auto-tuning under a split grid: the groups construct and tune different
# objects in different order and numbers; the tuner's collectives must stay
# within each group (world collectives would pair up across groups).
#
import sys

sys.argv.append("--auto-tune")

import gpt as g
import numpy as np
import os
import shutil

work = os.path.join(os.environ.get("WORK_DIR", "tests_work_dir"), "auto_tune")
if g.rank() == 0:
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
g.barrier()
os.chdir(work)

full_grid = g.grid([8, 8, 8, 8], g.double)
# groups of half the ranks if the last direction splits evenly (e.g. --mpi 1.1.2.2: 2 groups of 2), else one group
mpi = list(full_grid.mpi)
if mpi[3] % 2 == 0:
    mpi[3] //= 2
split_grid = full_grid.split(mpi, full_grid.fdimensions)
group = split_grid.srank
g.message(f"{split_grid.sranks} group(s) of {split_grid.Nprocessors} rank(s)")


def scale_stencil(lat, factor):
    # dst = factor * src through a local matrix stencil
    return g.local_stencil.matrix(
        lat, [(0, 0, 0, 0)], [{"target": 0, "accumulate": -1, "weight": factor, "factor": [(1, 0, 0)]}]
    )


def run(group):
    rng = g.random(f"auto_tune-{group}")
    if group == 0:
        # one object of one kind
        X = g.mcolor(split_grid)
        rng.cnormal(X)
        V = g(g.matrix.exp(0.5 * (X - g.adj(X))))
        eps2 = g.norm2(V * g.adj(V) - g.identity(V)) / g.norm2(V)
        assert eps2 < 1e-25, eps2
        return [V]
    # several objects, constructed and tuned in a different order
    res = []
    for k in range(3):
        v = g.mcolor(split_grid)
        rng.cnormal(v)
        st = scale_stencil(v, 1.0 + k)
        dst = g.lattice(v)
        st(dst, v)
        assert g.norm2(dst - (1.0 + k) * v) < 1e-20 * g.norm2(v)
        assert isinstance(st.at_tuned_params["params"], (tuple, list)) and len(st.at_tuned_params["params"]) == 2
        res.append(dst)
    return res


# first pass tunes (fresh cache), second pass loads the tuned results
for npass in range(2):
    r1 = run(group)
    full_grid.barrier()
    g.message(f"pass {npass}: group {group} done")

# tuned results are files written by the group roots
assert len([f for f in os.listdir(".gpt_auto_tune") if f.endswith(".json")]) > 0 if split_grid.processor == 0 else True

g.message("Test passed")
