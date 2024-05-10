#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Mattia Bruno     2020
#
# Load openqcd fields, apply open bc dslash, compare results
#
import os
import sys
import gpt as g
import numpy as np


# fdimensions
def fdimensions_from_openqcd(fdim):
    """
    Convert openqcd's coordinate ordering to our's: T X Y Z -> X Y Z T
    """
    assert len(fdim) == 4
    return [int(fdim[(i + 1) % 4]) for i in range(4)]


def fdimensions_to_openqcd(fdim):
    """
    Convert our coordinate ordering to openqcd's:  X Y Z T -> T X Y Z
    """
    assert len(fdim) == 4
    return [int(fdim[(i - 1) % 4]) for i in range(4)]


# helper functions
def fields_agree_openqcd(ref, res):
    """
    Implements the correctness check as it is used in openqcd, sarchive.c:614
    """
    if isinstance(ref, g.lattice) and isinstance(res, g.lattice):
        assert ref.grid.precision == g.double and res.grid.precision == g.double

    norm2_ref = g.norm2(ref) if isinstance(ref, g.lattice) else ref
    norm2_res = g.norm2(res) if isinstance(res, g.lattice) else res

    diff = abs(norm2_res - norm2_ref)
    tol = 64.0 * (norm2_ref + norm2_res) * sys.float_info.epsilon

    g.message(
        f"openqcd residual check: reference = {norm2_ref:25.20e}, result = {norm2_res:25.20e}, difference = {diff:25.20e}, tol = {tol:25.20e} -> check {'passed' if diff <= tol else 'failed'}"
    )
    return diff <= tol


def fields_agree(ref, res, tol):
    """
    Implements the standard correctness check between two fields with their relative deviation
    """
    norm2_ref = g.norm2(ref) if isinstance(ref, g.lattice) else ref
    norm2_res = g.norm2(res) if isinstance(res, g.lattice) else res

    if isinstance(ref, g.lattice) and isinstance(res, g.lattice):
        diff = g.norm2(res - ref)
    else:
        diff = abs(norm2_ref - norm2_res)

    rel_dev = diff / norm2_ref

    g.message(
        f"default residual check: reference = {norm2_ref:25.20e}, result = {norm2_res:25.20e},    rel_dev = {rel_dev:25.20e}, tol = {tol:25.20e} -> check {'passed' if rel_dev <= tol else 'failed'}"
    )
    return rel_dev <= tol


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

        # print(f"Check norm difference = {abs(np.sum(fld**2)-norm)}")

        fld = np.reshape(fld[:, :, :, :, :, 0] + 1j * fld[:, :, :, :, :, 1], tuple(lat) + (4, 3))
    return lat, norm, fld


def read_openqcd_fermion(fname, cb=None):
    """
    Read a spinor field written by openqcd into a vspincolor object.
    Extracts only the values on the sites corresponding to 'cb' if this parameter is set.
    """
    g.message(f"Reading spinor from file {fname} with checkerboard = {cb}")

    # read into numpy array
    lat, norm, fld = read_sfld(fname)

    # change the order in numpy and convert to gpt field
    # openqcd in memory, slow to fast: t x y z,
    # we in memory, slow to fast: t z y x,
    # -> need to swap x and z
    fld = np.swapaxes(fld, 1, 3)
    grid = g.grid(fdimensions_from_openqcd(lat), g.double)
    field = g.vspincolor(grid)
    field[:] = fld.flatten(order="C")
    norm2_file = norm[0]
    norm2_field = g.norm2(field)

    assert fields_agree_openqcd(norm2_file, norm2_field)
    assert fields_agree(norm2_file, norm2_field, g.double.eps)

    if not cb:
        return field

    assert cb in [g.even, g.odd]
    cbgrid = g.grid(
        grid.gdimensions,
        grid.precision,
        g.redblack,
        parent=grid.parent,
        mpi=grid.mpi,
    )
    cbfield = g.vspincolor(cbgrid)
    g.pick_checkerboard(cb, cbfield, field)
    return cbfield


# setup silent rng, mute
g.default.set_verbose("random", False)
rng = g.random("openqcd_dslash")

# fermion operator params
wc_params = {
    "kappa": 0.13500,
    "csw_r": 1.978,
    "csw_t": 1.978,
    "cF": 1.3,
    "xi_0": 1,
    "nu": 1,
    "isAnisotropic": False,
    "boundary_phases": [1.0, 1.0, 1.0, 0.0],
}

# workdir
if "WORK_DIR" in os.environ:
    work_dir = os.environ["WORK_DIR"]
else:
    work_dir = "."

# openqcd: 0 = open, 3 = antiperiodic bc
bc = 0

# files depend on whether open bc or not
request_files = [
    f"gauge.bc_{bc}.bin",
    f"src.bc_{bc}.bin",
    f"dst_Dw_dble.bc_{bc}.bin",
    f"dst_Dwee_dble.bc_{bc}.bin",
    f"dst_Dweo_dble.bc_{bc}.bin",
    f"dst_Dwoe_dble.bc_{bc}.bin",
    f"dst_Dwoo_dble.bc_{bc}.bin",
    f"dst_Dwdag_dble.bc_{bc}.bin",
]

# request test files
for f in request_files:
    g.repository.load(f"{work_dir}/{f}", f"gpt://tests/qcd/openqcd_dslash/{f}")

# load fields from files
U = g.load(f"{work_dir}/gauge.bc_{bc}.bin")
src = read_openqcd_fermion(f"{work_dir}/src.bc_{bc}.bin")
dst_oqcd_M = read_openqcd_fermion(f"{work_dir}/dst_Dw_dble.bc_{bc}.bin")
dst_oqcd_Moo = read_openqcd_fermion(f"{work_dir}/dst_Dwoo_dble.bc_{bc}.bin", g.odd)
dst_oqcd_Mee = read_openqcd_fermion(f"{work_dir}/dst_Dwee_dble.bc_{bc}.bin", g.even)
dst_oqcd_Moe = read_openqcd_fermion(f"{work_dir}/dst_Dwoe_dble.bc_{bc}.bin", g.odd)
dst_oqcd_Meo = read_openqcd_fermion(f"{work_dir}/dst_Dweo_dble.bc_{bc}.bin", g.even)
dst_oqcd_Mdag = read_openqcd_fermion(f"{work_dir}/dst_Dwdag_dble.bc_{bc}.bin")

# construct methods of our dirac operators from the read in fields
dst_oqcd_Dhop = g.lattice(dst_oqcd_M)
dst_oqcd_Mdiag = g.lattice(dst_oqcd_M)
g.set_checkerboard(dst_oqcd_Dhop, dst_oqcd_Moe)
g.set_checkerboard(dst_oqcd_Dhop, dst_oqcd_Meo)
g.set_checkerboard(dst_oqcd_Mdiag, dst_oqcd_Mee)
g.set_checkerboard(dst_oqcd_Mdiag, dst_oqcd_Moo)

# create fermion operators
wc_ref = g.qcd.fermion.reference.wilson_clover(U, wc_params)
wc_fast = g.qcd.fermion.wilson_clover(U, wc_params)

# src fields on half grids
src_e, src_o = g.vspincolor(wc_fast.F_grid_eo), g.vspincolor(wc_fast.F_grid_eo)
g.pick_checkerboard(g.even, src_e, src)
g.pick_checkerboard(g.odd, src_o, src)

# create list of test cases
tests = {}
for wc_name, wc in [("fast", wc_fast), ("reference", wc_ref)]:
    tests[wc_name] = {
        "M": {
            "mat": wc,
            "src": src,
            "dst_gpt": g.vspincolor(wc.F_grid),
            "dst_oqcd": dst_oqcd_M,
        },
        "Mee": {
            "mat": wc.Mooee,
            "src": src_e,
            "dst_gpt": g.vspincolor(wc.F_grid_eo),
            "dst_oqcd": dst_oqcd_Mee,
        },
        "Moo": {
            "mat": wc.Mooee,
            "src": src_o,
            "dst_gpt": g.vspincolor(wc.F_grid_eo),
            "dst_oqcd": dst_oqcd_Moo,
        },
        "Mdiag": {
            "mat": wc.Mdiag,
            "src": src,
            "dst_gpt": g.vspincolor(wc.F_grid),
            "dst_oqcd": dst_oqcd_Mdiag,
        },
        "Moe": {
            "mat": wc.Meooe,
            "src": src_e,
            "dst_gpt": g.vspincolor(wc.F_grid_eo),
            "dst_oqcd": dst_oqcd_Moe,
        },
        "Meo": {
            "mat": wc.Meooe,
            "src": src_o,
            "dst_gpt": g.vspincolor(wc.F_grid_eo),
            "dst_oqcd": dst_oqcd_Meo,
        },
        "Dhop": {
            "mat": wc.Dhop,
            "src": src,
            "dst_gpt": g.vspincolor(wc.F_grid),
            "dst_oqcd": dst_oqcd_Dhop,
        },
        "Mdag": {
            # reference doesn't have Mdag
            "mat": g.adj(wc) if wc_name == "new" else g.gamma[5] * wc * g.gamma[5],
            "src": src,
            "dst_gpt": g.vspincolor(wc.F_grid),
            "dst_oqcd": dst_oqcd_Mdag,
        },
    }

# apply the gpt operator and compare with openqcd
for wc_name, wc in [("fast", wc_fast), ("reference", wc_ref)]:
    for method_name, data in tests[wc_name].items():
        # set dst to random
        rng.cnormal(data["dst_gpt"])

        # apply gpt operator (openqcd uses different gamma representation)
        data["dst_gpt"] = g(g.gamma[1] * data["mat"] * g.gamma[1] * data["src"])

        # report error
        g.message(f"Checking residuals for {wc_name:10s} operator: {method_name}")
        assert fields_agree_openqcd(data["dst_oqcd"], data["dst_gpt"])
        assert fields_agree(data["dst_oqcd"], data["dst_gpt"], g.double.eps)

g.message("All tests successful")
