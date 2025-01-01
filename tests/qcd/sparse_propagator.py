#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2025
#
import gpt as g
import numpy as np
import os


# workdir
if "WORK_DIR" in os.environ:
    work_dir = os.environ["WORK_DIR"]
else:
    work_dir = "."

# request test files
files = ["30_combined.zip"]
for f in files:
    g.repository.load(f"{work_dir}/{f}", f"gpt://tests/qcd/propagators/{f}")

cache_params = (16, 4)
quark = g.qcd.sparse_propagator.flavor(f"{work_dir}/30_combined/light.e", *cache_params)

coordinates = quark.sink_domain.coordinates
nsrc = quark.source_domain.sampled_sites

# test cache optimized sampler
rng = g.random("test")
to_sample = [[rng.uniform_int(min=0, max=nsrc - 1) for i in range(3)] for j in range(20)]
sampled = []
for s in g.qcd.sparse_propagator.cache_optimized_sampler([quark], to_sample):
    q0 = quark[s[0]]
    q1 = quark[s[1]]
    q2 = quark[s[2]]
    sampled.append(s)

# check that sampled is only a permutation of to_sample
to_sample = [s[0] + 1000 * s[1] + 1000000 * s[2] for s in to_sample]
sample = [s[0] + 1000 * s[1] + 1000000 * s[2] for s in sampled]
for s in sample:
    assert s in to_sample
    to_sample.pop(to_sample.index(s))
assert to_sample == []

# now perform self-consistency checks of quark
for i_src in range(nsrc):

    g.message(f"Self-consistency check for src={i_src}")
    # get src-to-src propagator as numpy matrices
    src_src = quark(np.arange(nsrc), i_src)

    # get embedding in full lattice to test
    full = quark.sink_domain.sdomain.promote(quark[i_src])

    # test that src_src is consistent with quark[i_src] which embeds all into the sparse domain
    for i_snk in range(nsrc):
        eps2 = g.norm2(full[coordinates[i_snk]] - g.mspincolor(src_src[i_snk]))
        assert eps2 == 0.0

    # now remove elements
    mask = quark.source_mask()
    remove = [3, 8, 11]
    sparsened = g(mask * quark[i_src, remove])

    slice = quark.sink_domain.sdomain.slice(sparsened, 3)
    for i_snk in range(nsrc):
        if i_snk in remove:
            continue
        slice[coordinates[i_snk, 3]] -= g.mspincolor(src_src[i_snk])
    for s in slice:
        eps2 = g.norm2(s)
        assert eps2 < 1e-13
