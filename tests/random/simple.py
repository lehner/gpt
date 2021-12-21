#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

grid_dp = g.grid([8, 4, 4, 4], g.double)
grid_sp = g.grid([8, 4, 4, 4], g.single)
grid_large_dp = g.grid([8, 8, 4, 8], g.double)

rng = g.random("block_seed_string_13")


for grid, prec, ref_plaquette, scale in [
    (grid_dp, 1e-28, -0.00014108397456619623, 10),
    (grid_sp, 1e-14, -0.00014108397456619623, 10),
    (grid_large_dp, 1e-28, 0.38723058417632267, 2),
]:
    U = g.qcd.gauge.random(grid, rng, scale=scale)
    comp_plaquette = g.qcd.gauge.plaquette(U)
    eps2 = abs(comp_plaquette - ref_plaquette) ** 2.0
    g.message(
        f"Computed plaquette {comp_plaquette} versus reference {ref_plaquette}: {eps2}"
    )
    assert eps2 < prec
    for i in range(4):
        test = g.norm2(g.adj(U[i]) * U[i] - g.qcd.gauge.unit(grid)[0]) / g.norm2(U[i])
        g.message(test)
        assert test < prec


rng = g.random("block_seed_string_13")
n = 10000
res = {}
for i in range(n):
    z = rng.zn()
    if z not in res:
        res[z] = 0
    res[z] += 1

g.message(res)


n = 10000
res = {}
for i in range(n):
    z = rng.zn(n=3)
    if z not in res:
        res[z] = 0
    res[z] += 1

g.message(res)


n = 10000
res = [0, 0, 0, 0, 0]
for i in range(n):
    z = rng.normal()
    res[0] += 1
    res[1] += z
    res[2] += z ** 2
    res[3] += z ** 3
    res[4] += z ** 4

g.message(res[1] / res[0], res[2] / res[0], res[3] / res[0], res[4] / res[0])

v = g.complex(grid_dp)
rng.normal(v)

test_sequence_comp = np.array(
    [
        v[0, 0, 0, 0].real,
        v[2, 0, 0, 0].real,
        v[0, 2, 0, 0].real,
        v[1, 3, 1, 3].real,
        v[3, 2, 1, 0].real,
    ]
)
# print([ v[0,0,0,0].real, v[2,0,0,0].real, v[0,2,0,0].real, v[1,3,1,3].real, v[3,2,1,0].real ])
test_sequence_ref = np.array(
    [
        -0.29101665386129116,
        -1.4591269443435488,
        -0.3641310411719848,
        -0.9454532383815435,
        0.4996115272362977,
    ],
    np.float64,
)

for x in test_sequence_comp:
    g.message(x)

err = np.linalg.norm(test_sequence_comp - test_sequence_ref)
assert err < 1e-14

g.message(err)

for i in range(1000):
    rng.normal(v)

test_sequence_comp = np.array(
    [
        v[0, 0, 0, 0].real,
        v[2, 0, 0, 0].real,
        v[0, 2, 0, 0].real,
        v[1, 3, 1, 3].real,
        v[3, 2, 1, 0].real,
    ]
)


for x in test_sequence_comp:
    g.message(x)

test_sequence_ref = np.array(
    [
        1.473846437649123,
        0.06134886004475955,
        -1.4849224560837744,
        -0.2316303634513769,
        -0.5309613807759392,
    ],
    np.float64,
)

err = np.linalg.norm(test_sequence_comp - test_sequence_ref)
assert err < 1e-14

assert rng.choice(["A", "B", "C"], 10) == [
    "C",
    "C",
    "A",
    "C",
    "C",
    "B",
    "A",
    "B",
    "C",
    "A",
]
assert np.all(rng.choice(np.array([1, 2, 3]), 5) == np.array([2, 3, 3, 3, 1]))
g.message("All tests passed")
