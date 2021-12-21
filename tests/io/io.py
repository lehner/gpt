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

# create a sparse sub-domain and a sparse lattice S with 1% of points
sdomain = g.domain.sparse(
    U[0].grid,
    rng.choice(
        g.coordinates(U[0]), int(0.01 * U[0].grid.gsites / U[0].grid.Nprocessors)
    ),
)

# test sparse domain
S = sdomain.lattice(U[0].otype)
sdomain.project(S, U[0])
U0prime = g.lattice(U[0])
U0prime[:] = 0
sdomain.promote(U0prime, S)
assert (
    np.linalg.norm(U0prime[sdomain.local_coordinates] - U[0][sdomain.local_coordinates])
    < 1e-14
)

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
        "sdomain": sdomain,
        "S": S,
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

eps2 = g.norm2(res["S"] - S)
g.message("Test sparse field restore:", eps2)
assert eps2 < 1e-25

# check load sparse lattice
U0prime2 = g.lattice(U[0])
U0prime2[:] = 0
res["sdomain"].promote(U0prime2, res["S"])
eps2 = g.norm2(U0prime - U0prime2)
g.message("Test sparse domain restore:", eps2)
assert eps2 < 1e-25

# check load out2 with fixed mpi
res = g.load(f"{work_dir}/out2", paths="/U/*")
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

assert g.corr_io.count(f"{work_dir}/head.dat") == 1


# NERSC
fn = f"{work_dir}/ckpoint.0000"
g.save(fn, U, g.format.nersc())
U_prime = g.load(fn)
assert len(U_prime) == len(U)
for u_prime, u in zip(U_prime, U):
    eps = (g.norm2(u_prime - u) / g.norm2(u)) ** 0.5
    g.message(f"Test NERSC IO: {eps}")
    assert eps < 1e-14
