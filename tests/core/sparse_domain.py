#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020 Mattia Bruno 2026
#
# Desc.: Test small features of sparse domains
#
import gpt as g
import numpy as np
import os

grid = g.grid([16,16,16,32], g.double)

position_seed = 'sparse-domain'
rng = g.random(position_seed)

nsnk = 128
all_positions = np.array([
    [rng.uniform_int(min=0, max=grid.fdimensions[i] - 1) for i in range(4)] for j in range(nsnk)
], dtype=np.int32)

nsnk_local = nsnk // grid.Nprocessors

sdomain = g.domain.sparse(grid, all_positions[g.rank() * nsnk_local: (g.rank()+1) * nsnk_local])

assert abs(g.sum(sdomain.weight()) - nsnk) < 1e-16
assert abs(g.sum(sdomain.kernel.cached_one_mask()) - nsnk) < 1e-16

nsrc = 16
N = nsnk - nsrc
subset = all_positions[nsrc:]

sdomain2 = sdomain.restricted(subset)

g.message(f'Restricted sparse domain from {g.sum(sdomain.weight())} points down to {g.sum(sdomain2.weight())}')

assert abs(g.sum(sdomain2.weight()) - N) < 1e-16
assert abs(g.sum(sdomain2.kernel.cached_one_mask()) - N) < 1e-16

# sparse domain irregularly distributed over nodes

def gsum(lat):
    idx = g.real(lat.grid)
    idx[:] = 0
    return g.indexed_sum(lat, idx, 1)[0]

np.random.seed(46)

def random_partition_sizes(N, M, spread=2):
    base = N // M
    sizes = [base] * M
    if M==1:
        return sizes
    
    # distribute remainder randomly
    for _ in range(N - base * M):
        sizes[np.random.randrange(M)] += 1

    # random perturbations
    for _ in range(spread * M):
        i, j = np.random.choice(M, size=2, replace=False)

        if sizes[i] > 1:
            sizes[i] -= 1
            sizes[j] += 1

    return sizes

nsnk_local = random_partition_sizes(nsnk, g.ranks(), spread=int(nsnk*0.1))
sdomain3 = g.domain.sparse(grid, all_positions[g.rank() * nsnk_local[g.rank()]: (g.rank()+1) * nsnk_local[g.rank()]])

assert abs(gsum(sdomain.weight()) - nsnk) < 1e-16
assert abs(gsum(sdomain.kernel.cached_one_mask()) - nsnk) < 1e-16