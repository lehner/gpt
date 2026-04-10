#!/usr/bin/env python3
#
# Authors: Christoph Lehner
#
import gpt as g
import numpy as np
import sys, os

# parameters
tau = g.default.get_float("--tau", 1.0)
root = g.default.get("--root", None)
seed = g.default.get("--seed", "hmc-pure-gauge")
n = g.default.get_int("--n", 1000)

# grid
grid = g.grid([4, 4, 4, 8], g.double)
rng = g.random(seed)

# load state / initialize state
A = [g.real(grid) for _ in range(4)]
rng.normal_element(A)

fn_try = None
i0 = 0
for i in range(0, n):
    fn = f"{root}/ckpoint_lat.A.{i}"
    if os.path.exists(fn):
        fn_try = fn
        i0 = i + 1

if fn_try is not None:
    rng = g.random(fn_try)
    A0 = g.load(fn_try)
    for mu in range(4):
        A[mu] @= A0[mu]

a_qed = g.qcd.gauge.action.non_compact.qed_l(grid)

p = 2*np.pi * np.array([1,0,0,0]) / np.array(grid.gdimensions)
prop = g(g.exp_ixp(p) * a_qed.propagator()[0][0])
cor_ref = g.slice(prop, 3)

# production
f = open("c1.draw.dat", "wt")
for i in range(i0, n):
    g.message(f"Trajectory {i}")

    a_qed.draw(A, rng)

    prop = g(g.exp_ixp(p) * g.correlate(A[0], A[0]))
    cor = g.slice(prop, 3)

    f.write(f"{i} {cor[1].real}\n")
    f.flush()

    o = g.corr_io.writer(f"{root}/{i}.draw.dat")
    o.write("ref", cor_ref)
    o.write("cor", cor)
    o.close()
    
    # g.save(f"{root}/ckpoint_lat.A.{i}", A)

    # if g.rank() == 0:
    #     flog = open(f"{root}/ckpoint_lat.A.{i}.log", "wt")
    #     flog.write(f"dH {dH}\n")
    #     flog.close()

    g.barrier()

f.close()
