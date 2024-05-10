#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Calculate HVP connected diagram with A2A method
#
import gpt as g
import sys

# show available memory
g.mem_report()

# parameters
fn = g.default.get("--params", "params.txt")
params = g.params(fn, verbose=True)

# load configuration
U = params["config"]
L = U[0].grid.fdimensions

# show available memory
g.mem_report()

# quark flavors
l_exact = params["light_exact"](U)
l_sloppy = params["light_sloppy"](U, l_exact)

# eigenvectors and values, may be MG
l_sloppy_eigensystem = params["light_sloppy_eigensystem"](U, l_sloppy)

# show available memory
g.mem_report()

# sloppy solver
l_sloppy_inverter = params["light_sloppy_inverter"](U, l_sloppy_eigensystem)

l_low_inverter = params["light_low_inverter"](U, l_sloppy_eigensystem)

# propagators
prop_l_low = l_sloppy.propagator(l_low_inverter)
prop_l_sloppy = l_sloppy.propagator(l_sloppy_inverter)
prop_l_approx = l_exact.propagator(params["light_approx_inverter"](U, l_sloppy_inverter))
prop_l_exact = l_exact.propagator(params["light_exact_inverter"](U, l_sloppy_inverter))
prop_l_corrected_low = l_exact.propagator(params["light_corrected_low_inverter"](U, l_low_inverter))
del l_sloppy_inverter

# random number generator
rng = params["rng"]

# AMA strategy
# (exact - sloppy)_positions_exact + (sloppy - low)_positions_sloppy + low_volume_average

# source positions
source_positions_sloppy = params["source_positions_sloppy"](rng, L)
source_positions_exact = params["source_positions_exact"](rng, L)

# output
output = params["output"]
output_correlator = params["output_correlator"]

# save positions
pos_archive = {
    "source_positions_sloppy": source_positions_sloppy,
    "source_positions_exact": source_positions_exact,
}
output.write(pos_archive)
g.message("Source Positions:")
g.message(pos_archive)

# correlators
operators = params["operators"]
correlators = params["correlators"]


def contract(pos, prop, tag, may_save_prop=True):
    t0 = pos[3]
    prop_tag = "%s/%s" % (tag, str(pos))

    # save propagators
    if params["save_propagators"] and may_save_prop:
        output.write({prop_tag: prop})
        output.flush()

    # create and save correlators
    for op_snk, op_src in correlators:
        G_snk = operators[op_snk]
        G_src = operators[op_src]
        corr = g.slice(g.trace(G_src * g.gamma[5] * g.adj(prop) * g.gamma[5] * G_snk * prop), 3)
        corr = corr[t0:] + corr[:t0]

        corr_tag = "%s/snk%s-src%s" % (prop_tag, op_snk, op_src)
        output_correlator.write(corr_tag, corr)
        g.message("Correlator %s\n" % corr_tag, corr)


# calculate correlators for exact positions
vol3d = (
    l_exact.U_grid.fdimensions[0] * l_exact.U_grid.fdimensions[1] * l_exact.U_grid.fdimensions[2]
)
# vol3d = 8
# spacing = [48, 48, 48, 192]

source_time_slices = 2
full_time = l_exact.U_grid.fdimensions[3]
assert full_time % source_time_slices == 0
sparse_time = full_time // source_time_slices


for pos in source_positions_exact:
    srcD = g.mspincolor(l_exact.U_grid)
    srcD[:] = 0

    # create time-sparsened source
    sign_of_slice = [rng.zn(n=2) for i in range(source_time_slices)]
    pos_of_slice = [
        [pos[i] if i < 3 else (pos[i] + j * sparse_time) % full_time for i in range(4)]
        for j in range(source_time_slices)
    ]
    g.message(f"Signature: {pos} -> {pos_of_slice} with signs {sign_of_slice}")
    for i in range(source_time_slices):
        # srcD += g.create.point(g.lattice(srcD), pos_of_slice[i]) * sign_of_slice[i]
        srcD += g.create.wall.z2(g.lattice(srcD), pos_of_slice[i][3], rng) * (
            sign_of_slice[i] / vol3d**0.5
        )

    # g.create.wall.z2(srcD, pos[3], rng)
    # g.create.sparse_grid.zn(srcD, pos, spacing, rng, 2)
    # srcD /= vol3d**0.5
    srcF = g.convert(srcD, g.single)

    # exact_sloppy
    prop_exact = g.eval(prop_l_exact * srcD)
    prop_approx = g.eval(prop_l_approx * srcD)
    prop_low = g.eval(prop_l_low * srcF)
    # prop_corrected_low = g.eval(prop_l_corrected_low * srcD)
    for i in range(source_time_slices):
        contract(pos_of_slice[i], g.eval(sign_of_slice[i] * prop_exact), "exact")
        contract(pos_of_slice[i], g.eval(sign_of_slice[i] * prop_approx), "approx")
        contract(pos_of_slice[i], g.eval(sign_of_slice[i] * prop_low), "low")
        # contract(pos_of_slice[i], g.eval( sign_of_slice[i] * prop_corrected_low ), "clow")


sys.exit(0)

# fine_n = promote * coarse_n
# sum_x fine_n^T fine_m = sum_x coarse_n^T promote^T_x promote_x coarse_m

# calculate correlators for sloppy positions
for pos in source_positions_sloppy:
    # sloppy
    src = g.mspincolor(l_sloppy.U_grid)
    g.create.point(src, pos)
    contract(pos, g.eval(prop_l_sloppy * src), "sloppy")

# release propagators (and their references to eigenvector data)
del prop_l_exact
del prop_l_sloppy

# separate eigensystem
basis, cevec, evals = l_sloppy_eigensystem
del l_sloppy_eigensystem

# create basis for a2a vectors
a2a = params["a2a"](U, l_sloppy)
if a2a is None:
    sys.exit(0)

a2a_coarse_grid = params["a2a_coarse_grid"](cevec[0].grid)
tmpf = g.lattice(basis[0])

# left and right vectors
a2a_left, a2a_right = params["a2a_vectors"](a2a)

# keep vectors for sanity check
test_left = a2a_left(g.block.promote(cevec[0], tmpf, basis))
test_right = a2a_right(g.block.promote(cevec[0], tmpf, basis))

# create basis based on left vector
t0 = g.time()
a2a_basis = [
    a2a_left(g.block.promote(cevec[i], tmpf, basis)) for i in range(params["a2a_coarse_nbasis"])
]
t1 = g.time()
for i in range(params["a2a_coarse_basis_ortho_steps"]):
    g.block.orthonormalize(a2a_coarse_grid, a2a_basis)
t2 = g.time()
g.message(
    "Creating the A2A coarse basis took",
    t1 - t0,
    "s and",
    t2 - t1,
    "s for orthonormalization",
)

# now create and compress v and w vectors
a2a_cleft, a2a_cright = [], []
for i in range(len(cevec)):
    t0 = g.time()
    a2a_cleft.append(
        g.block.project(
            g.vcomplex(a2a_coarse_grid, len(a2a_basis)),
            a2a_left(g.block.promote(cevec[i], tmpf, basis)),
            a2a_basis,
        )
    )
    a2a_cright.append(
        g.block.project(
            g.vcomplex(a2a_coarse_grid, len(a2a_basis)),
            a2a_right(g.block.promote(cevec[i], tmpf, basis)),
            a2a_basis,
        )
    )
    t1 = g.time()
    cevec[i] = None  # release memory
    g.message("Create compressed left/right vectors %d in %g s" % (i, t1 - t0))
del basis
g.mem_report()

# complete sanity check
tmpf = g.vspincolor(l_sloppy.U_grid)
g.block.promote(a2a_cleft[0], tmpf, a2a_basis)
g.message("Test left[0]", g.norm2(tmpf - test_left) / g.norm2(tmpf + test_left))
g.block.promote(a2a_cright[0], tmpf, a2a_basis)
g.message("Test right[0]", g.norm2(tmpf - test_right) / g.norm2(tmpf + test_right))

# save vectors
if params["a2a_save_vectors"]:
    output.write({"a2a_basis": a2a_basis, "a2a_cleft": a2a_cleft, "a2a_cright": a2a_cright})

# create coarse approximation matrix
prop_a2a = (
    g.algorithms.approx.coarse_modes(
        a2a_basis, a2a_basis, a2a_cleft, a2a_cright, evals, lambda x: 1.0 / x
    )()
    * params["a2a_preconditioner"]
)

# calculate correlators for sloppy positions
for pos in source_positions_sloppy + source_positions_exact:
    # sloppy
    src = g.mspincolor(l_sloppy.U_grid)
    g.create.point(src, pos)
    contract(pos, g.eval(prop_a2a * src), "low", False)

# cleanup
params = None
output.close()
sys.exit(0)
