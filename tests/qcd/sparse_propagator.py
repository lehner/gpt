#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2025
#
import gpt as g
import numpy as np
import os, sys

# adjust mpi setup
for i in range(len(sys.argv)):
    if sys.argv[i] == "--mpi":
        mpi = [int(x) for x in sys.argv[i + 1].split(".")]
        if len(mpi) == 4:
            mpi = list(reversed(mpi))
        sys.argv[i + 1] = ".".join([str(x) for x in mpi])

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

for tag in ["e", "s", "ems", "l", "sml"]:

    g.message(
        f"""

    Test {tag}

    """
    )
    quark = g.qcd.sparse_propagator.flavor(f"{work_dir}/30_combined/light.{tag}", *cache_params)

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
        remove = [x for x in [1, 8, 11] if x < nsrc]
        sparsened = g(mask * quark[i_src, remove])

        slice = quark.sink_domain.sdomain.slice(sparsened, 3)
        for i_snk in range(nsrc):
            if i_snk in remove:
                continue
            slice[coordinates[i_snk, 3]] -= g.mspincolor(src_src[i_snk])
        for s in slice:
            eps2 = g.norm2(s)
            assert eps2 < 1e-13


# extra tests between tags
quark_sml = g.qcd.sparse_propagator.flavor(f"{work_dir}/30_combined/light.sml", *cache_params)
quark_s = g.qcd.sparse_propagator.flavor(f"{work_dir}/30_combined/light.s", *cache_params)
quark_l = g.qcd.sparse_propagator.flavor(f"{work_dir}/30_combined/light.l", *cache_params)
qs_ec = quark_s.sink_domain.sdomain.unique_embedded_coordinates(quark_l.sink_domain.coordinates)
ql_ec = quark_l.sink_domain.sdomain.unique_embedded_coordinates(quark_l.sink_domain.coordinates)
qsml_ec = quark_sml.sink_domain.sdomain.unique_embedded_coordinates(quark_l.sink_domain.coordinates)

assert len(quark_l.sink_domain.coordinates) == quark_l.sink_domain.sampled_sites
assert quark_sml.sink_domain.sampled_sites == quark_l.sink_domain.sampled_sites
assert len(quark_sml.sink_domain.coordinates) == quark_sml.sink_domain.sampled_sites

for i in range(quark_sml.source_domain.sampled_sites):
    qs_i = quark_s[i]
    ql_i = quark_l[i]
    qsml_i = quark_sml[i]
    for j in range(quark_l.sink_domain.sampled_sites):
        eps = qs_i[qs_ec[j]] - ql_i[ql_ec[j]]
        # for test data same solver is used for sloppy and low
        assert g.norm2(eps) == 0.0
        assert g.norm2(qsml_i[qsml_ec[j]]) == 0.0
    x = quark_sml.sink_domain.sdomain.slice(qsml_i, 3)
    for y in x:
        assert g.norm2(y) < 1e-13

# now test sink-conformal mapping;
# sloppy has more sink sites than low,
# by mapping them jointly we restrict sloppy
# to low's sink domain automatically such that
# they can be used in a conformal manner
assert not np.array_equal(
    quark_s.sink_domain.sdomain.kernel.local_coordinates,
    quark_l.sink_domain.sdomain.kernel.local_coordinates,
)
quark_s, quark_l = g.qcd.sparse_propagator.flavor(
    [f"{work_dir}/30_combined/light.s", f"{work_dir}/30_combined/light.l"], *cache_params
)
assert np.array_equal(
    quark_s.sink_domain.sdomain.kernel.local_coordinates,
    quark_l.sink_domain.sdomain.kernel.local_coordinates,
)

for i in range(quark_sml.source_domain.sampled_sites):
    qs_i = quark_s[i]
    ql_i = quark_l[i]
    assert g.norm2(qs_i - ql_i) < 1e-13
