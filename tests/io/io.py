#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys
import os
import random
import cgpt

# workdir
if "WORK_DIR" in os.environ:
    work_dir = os.environ["WORK_DIR"]
else:
    work_dir = "."

# load configuration
# U = g.load("/hpcgpfs01/work/clehner/configs/32IDfine/ckpoint_lat.200")
# assert abs(g.qcd.gauge.plaquette(U) - float(U[0].metadata["PLAQUETTE"])) < 1e-9

# Show metadata of field
# g.message("Metadata", U[0].metadata)
rng = g.random("test")
U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], g.double), rng)

# save in default gpt format
g.save(
    f"{work_dir}/out",
    {
        "va\nl": [
            0,
            1,
            3,
            "tes\n\0t",
            3.123456789123456789,
            1.123456789123456789e-7,
            1 + 3.1231251251234123413j,
        ],  # fundamental data types
        "np": g.coordinates(U[0].grid),  # write numpy array from root node
        "U": U,  # write list of lattices
    },
)

# save in custom gpt format with different mpi distribution of local views
g.save(
    f"{work_dir}/out2",
    {
        "val": [
            0,
            1,
            3,
            "test",
            3.123456789123456789,
            1.123456789123456789e-7,
            1 + 3.1231251251234123413j,
        ],  # fundamental data types
        "np": g.coordinates(U[0].grid),  # write numpy array from root node
        "U": U,  # write list of lattices
    },
    g.format.gpt(
        {
            "mpi": [
                2,
                2,
                2,
                1,
            ]  # save fields in 2 x 2 x 1 x 1 processor grid instead of --mpi grid
        }
    ),
)

#
# load function
#
# - g.load(fn)          loads everything in fn and creates new grids as needed
# - g.load(fn,{ "grids" : ..., "paths" :  ... })  both grids and paths are optional parameters and may be lists,
#                                                 grids are re-used when loading, paths restricts which items to load (allows for glob.glob syntax /U/*)
res = g.load(f"{work_dir}/out")

for i in range(4):
    eps2 = g.norm2(res["U"][i] - U[i])
    g.message("Test first restore of U[%d]:" % i, eps2)
    assert eps2 < 1e-25

res = g.load(f"{work_dir}/out2", {"paths": "/U/*"})
for i in range(4):
    eps2 = g.norm2(res["U"][i] - U[i])
    g.message("Test second restore of U[%d]:" % i, eps2)
    assert eps2 < 1e-25

# checkpointer save
ckpt = g.checkpointer(f"{work_dir}/ckpt")
alpha = 0.125
ckpt.save([U[0], alpha])

# checkpointer load
ckpt = g.checkpointer(f"{work_dir}/ckpt")
ckpt.grid = U[0].grid
alpha = 0.125
U0_test = g.lattice(U[0])
assert ckpt.load([U0_test, alpha])
assert abs(alpha - 0.125) < 1e-25
assert g.norm2(U0_test - U[0]) == 0.0

# corr-io
corr = [rng.normal().real for i in range(32)]
w = g.corr_io.writer(f"{work_dir}/head.dat")
w.write("test", corr)
w.close()

r = g.corr_io.reader(f"{work_dir}/head.dat")
assert "test" in r.glob("*")
for i in range(len(corr)):
    assert abs(r.tags["test"][i] - corr[i]) == 0.0

# sys.exit(0)


# # Calculate Plaquette
# g.message(g.qcd.gauge.plaquette(U))

# # Precision change
# Uf = g.convert(U, g.single)
# g.message(g.qcd.gauge.plaquette(Uf))

# Uf0 = g.convert(U[0], g.single)
# g.message(g.norm2(Uf0))

# del Uf0
# g.mem_report()

# # Slice
# x = g.sum(Uf[0])

# print(x)

# grid = g.grid([4, 4, 4, 4], g.single)
# gr = g.complex(grid)

# gr[0, 0, 0, 0] = 2
# gr[1, 0, 0, 0] = 3

# gride = g.grid([4, 4, 4, 4], g.single, g.redblack)
# gre = g.complex(gride)
# g.pick_checkerboard(g.even, gre, gr)
# gre[2, 0, 0, 0] = 4
# g.set_checkerboard(gr, gre)
# g.mem_report()


# print(gre)

# gre.checkerboard(g.odd)

# print(gre)


# sys.exit(0)

# # Calculate U^\dag U
# u = U[0][0, 1, 2, 3]

# v = g.vcolor([0, 1, 0])

# g.message(g.adj(v))
# g.message(g.adj(u) * u * v)


# gr = g.grid([2, 2, 2, 2], g.single)
# g.message(g.mspincolor(gr)[0, 0, 0, 0] * g.vspincolor(gr)[0, 0, 0, 0])

# g.message(g.trace(g.mspincolor(gr)[0, 0, 0, 0]))

# # Expression including numpy array
# r = g.eval(u * U[0] + U[1] * u)
# g.message(g.norm2(r))

# # test inner and outer products
# v = g.vspincolor([[0, 0, 0], [0, 0, 2], [0, 0, 0], [0, 0, 0]])
# w = g.vspincolor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]])
# xx = v * g.adj(w)
# g.message(xx[1][3][2][0])
# g.message(xx)
# g.message(g.adj(v) * v)

# g.message(g.transpose(v) * v)

# u += g.adj(u)
# g.message(u)


# v = g.vspincolor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
# l = g.vspincolor(gr)
# l[:] = 0
# l[0, 0, 0, 0] = v

# g.message(l)

# for mu in [0, 1, 2, 3, 5]:
#     for nu in [0, 1, 2, 3, 5]:
#         g.message(
#             mu,
#             nu,
#             g.norm2(g.gamma[mu] * g.gamma[nu] * l + g.gamma[nu] * g.gamma[mu] * l)
#             / g.norm2(l),
#         )

# g.message(l)

# m = g.mspincolor(gr)
# m[0, 0, 0, 0] = xx
# m @= g.gamma[5] * m * g.gamma[5]
# g.message(m)
