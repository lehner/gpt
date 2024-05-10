#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Benchmark bandwidth of assignments
#
import gpt as g
import socket as s
import sys

n = g.default.get_int("--n", 10)
nwarm = 10

# mpi information
r = g.rank()
N = g.ranks()
hosts = []
for i in range(N):
    hosts.append(g.broadcast(i, f"{r}-{s.gethostname()}"))

# now create data that we can use to benchmark
grid = g.grid(g.default.get_ivec("--grid", [8, 8, 8, 16], 4), g.double)
lattice_host = g.vcolor(grid)
lattice_accelerator = g.vcolor(grid)
lattice_host[:] = 1
lattice_accelerator[:] = 1

# perform an operation to move data to accelerator or host
lattice_accelerator @= 2 * lattice_accelerator
lattice_host.mview()

# now create copy plans from each to each rank
local_coordinates = g.coordinates(lattice_host)

ngb_single = lattice_host.rank_bytes() / 1e9
ngb = ngb_single * n

g.message(f"Benchmark sending {ngb_single} GB from rank to rank {n} times")

for i in range(N):
    i_coordinates = g.broadcast(i, local_coordinates)

    for j in range(N):
        j_coordinates = g.broadcast(j, local_coordinates)

        if i != j:
            msg = ""

            for tag, lat in [("host", lattice_host), ("acc", lattice_accelerator)]:
                for use_communication_buffers in [False, True]:
                    plan = g.copy_plan(lat, lat)
                    plan.destination += lat.view[i_coordinates]
                    plan.source += lat.view[j_coordinates]

                    plan = plan(use_communication_buffers=use_communication_buffers)

                    info = plan.info()
                    for a in info:
                        for b in info[a]:
                            assert info[a][b]["blocks"] == 1

                    for l in range(nwarm):
                        plan(lat, lat)

                    t0 = g.time()
                    for l in range(n):
                        plan(lat, lat)
                    t1 = g.time()

                    msg = (
                        msg
                        + f"{ngb/(t1-t0):.3g} GB/s ({tag}->{tag},use_buf={use_communication_buffers})   "
                    )

            g.message(f"{hosts[i]} -> {hosts[j]} : {msg}")
