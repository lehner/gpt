#!/usr/bin/env python3
#
# Authors: Mattia Bruno 2020
#          Christoph Lehner 2020
#
import sys
import os
import gpt
import numpy

# workdir
if "WORK_DIR" in os.environ:
    work_dir = os.environ["WORK_DIR"]
else:
    work_dir = "."

# request test files
files = ["psrc-prop-0.field", "pion-corr.txt"]
for f in files:
    gpt.repository.load(f"{work_dir}/{f}", f"gpt://tests/io/qlat/{f}")

# load field
prop = gpt.load(f"{work_dir}/psrc-prop-0.field")
gpt.message("Grid from qlat propagator =", prop.grid)

# calculate correlator
corr_pion = gpt.slice(gpt.trace(gpt.adj(prop) * prop), 3)

# load reference
with open(f"{work_dir}/pion-corr.txt", "r") as f:
    txt = f.readlines()

# read lines corresponding to real part of time slices and
# check difference w.r.t. what we have loaded above
for i in range(8):
    ref = float(txt[1 + i * 2].split(" ")[-1][:-1])
    diff = abs(ref - corr_pion[i].real)
    assert diff < 1e-7  # propagator was computed in single precision
    gpt.message("Time slice %d difference %g" % (i, diff))

gpt.message("Test successful")
