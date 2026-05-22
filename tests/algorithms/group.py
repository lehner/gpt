#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2026
#
import gpt as g

# test number of operators in t1u irreps
minmom = [
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (2, 0, 0),
    (2, 1, 0),
    (2, 1, 1),
    (2, 2, 0),
    (2, 2, 1),
    (3, 0, 0),
    (3, 1, 0),
    (3, 1, 1),
    (3, 2, 1),
    (7, 5, 3),
]
for i in range(3):
    rops = [len(g.algorithms.group.full_octahedral.t1u_all(mm, i)) for mm in minmom]
    assert rops == [1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 3, 3]
    g.message(f"Checked T1-_{i}")
