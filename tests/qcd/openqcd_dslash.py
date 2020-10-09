#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Mattia Bruno     2020
#
# Load openqcd fields, apply open bc dslash, compare results
#
import os
import gpt as g
import numpy as np

# helper functions
def read_sfld(fname):
    """
    Read a spinor field written by openqcd into a numpy array.
    Function was written by Mattia Bruno.
    """
    with open(fname, "rb") as f:
        b = f.read(4 * 5)
        tmp = np.frombuffer(b, dtype=np.int32)
        lat = tmp[0:4]
        assert tmp[4] == 0

        b = f.read(8)
        norm = np.frombuffer(b, dtype=np.float64)

        vol = np.prod(lat)
        b = f.read(vol * 12 * 2 * 8)

        fld = np.frombuffer(b, dtype=np.float64)
        fld = np.reshape(fld, tuple(lat) + (12, 2))

        print(f"Check norm difference = {abs(np.sum(fld**2)-norm)}")

        fld = np.reshape(
            fld[:, :, :, :, :, 0] + 1j * fld[:, :, :, :, :, 1], tuple(lat) + (4, 3)
        )
    return lat, norm, fld


def read_openqcd_fermion(fname):
    """
    Read a spinor field written by openqcd into a vspincolor object.
    """
    # read into numpy array
    lat, norm, fld = read_sfld(fname)

    # convert numpy to gpt field
    field = g.vspincolor(g.grid(g.util.fdimensions_from_openqcd(lat), g.double))
    field[:] = 0.0
    X, Y, Z, T = field.grid.fdimensions
    for t in range(T):
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    field[x, y, z, t] = g.vspincolor(fld[t, x, y, z])

    # sanity check
    norm_gpt = g.norm2(field)
    diff = abs(norm[0] - norm_gpt)
    eps = (diff / norm[0]) ** 0.5
    g.message(f"Norms: file = {norm[0]}, gpt = {norm_gpt}, diff = {diff}, eps = {eps}")
    assert eps < 1e-15

    return field


# workdir
if "WORK_DIR" in os.environ:
    work_dir = os.environ["WORK_DIR"]
else:
    work_dir = "."

# request test files
for f in ["field_gauge.bin", "field_src.bin", "field_dst.bin"]:
    g.repository.load(f"{work_dir}/{f}", f"gpt://tests/qcd/openqcd_dslash/{f}")

# load fields from files
U = g.load(f"{work_dir}/field_gauge.bin")
src = read_openqcd_fermion(f"{work_dir}/field_src.bin")
dst_oqcd = read_openqcd_fermion(f"{work_dir}/field_dst.bin")

# fermion operator
w = g.qcd.fermion.reference.wilson_clover(
    U,
    {
        "kappa": 0.13500,
        "csw_r": 1.978,
        "csw_t": 1.978,
        "cF": 1.3,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 0.0],  # this triggers open bc
    },
)

# apply gpt operator (openqcd uses different representation)
dst_gpt = g(g.gamma[1] * w * g.gamma[1] * src)

# report error
eps = (g.norm2(dst_gpt - dst_oqcd) / g.norm2(dst_oqcd)) ** 0.5
g.message(
    f"Test for matching operators {'passed' if eps <= 1e-15 else 'failed'}. eps = {eps}, tol = {1e-15}"
)
assert eps <= 1e-15
