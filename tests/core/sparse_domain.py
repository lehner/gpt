#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020 Mattia Bruno 2026
#
# Desc.: Test small features of sparse domains
#
import gpt as g
import numpy as np

grid = g.grid([8,8,8,8], g.double)

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
