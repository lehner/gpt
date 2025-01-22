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

files = ["test_prop_single.h5", "test_prop_double.h5"]
for f in files:
    g.repository.load(f"{work_dir}/{f}", f"gpt://tests/io/hdf5/{f}")

fd = g.load(f"{work_dir}/test_prop_double.h5", paths=["/diracpropagator/data"])
fs = g.load(f"{work_dir}/test_prop_single.h5", paths=["/diracpropagator/data"])

eps2 = g.norm2(g.convert(fd, g.single) - fs) / g.norm2(fs)
g.message(f"Single/double test: {eps2}")
assert eps2 < 1e-6

fprng = g.random("fp")
fp = np.array([g.inner_product(fprng.cnormal(g.lattice(fd)), fd) for i in range(8)])
ref_fp = np.array(
    [
        (-0.17193155920699343 + 0.979208077143875j),
        (0.2656265720403854 - 2.5119332658481435j),
        (0.9409108134704344 + 0.22593234908324247j),
        (-0.32534285045233957 - 0.43625690977492365j),
        (-1.3933479193502183 + 0.27439917194141367j),
        (1.5985342122990607 - 0.18334815996794449j),
        (0.7117903364384113 - 0.28037398097521676j),
        (-0.12953818164169206 - 0.14946957244148398j),
    ]
)

eps = np.linalg.norm(fp - ref_fp)
g.message(f"Fingerprint test: {eps}")
assert eps < 1e-13
