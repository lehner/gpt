#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Production code to generate coarse-grid eigenvectors using existing
# fine-grid basis vectors
#
import gpt as g

# show available memory
g.mem_report()

# parameters
fn = g.default.get("--params", "params.txt")
params = g.params(fn, verbose=True)

# load configuration
U = params["config"]

# show available memory
g.mem_report()

# fermion
q = params["fmatrix"](U)

# load basis vectors
nbasis = params["nbasis"]
fg_basis, fg_cevec, fg_feval = g.load(
    params["basis"],
    grids=q.Mpc.grid[0],
    nmax=nbasis,
    advise_basis=g.infrequent_use,
    advise_cevec=g.infrequent_use,
)

# memory info
g.mem_report()

# norms
for i in range(nbasis):
    g.message("Norm2 of basis[%d] = %g" % (i, g.norm2(fg_basis[i])))

for i in range(nbasis):
    g.message("Norm2 of cevec[%d] = %g" % (i, g.norm2(fg_cevec[i])))

g.mem_report()

# prepare and test basis
basis = []
assert nbasis > 0
b = g.block.map(fg_cevec[0].grid, fg_basis)
for i in range(nbasis):
    basis.append(
        g.vspincolor(q.Mpc.grid[0])
    )  # don't advise yet, let it be first touched on accelerator
    g.message(i)
    if i < params["nbasis_on_host"]:
        g.message("marked as infrequent use")
        # basis[i].advise( g.infrequent_use )

    basis[i] @= b.promote * fg_cevec[i]
    g.algorithms.eigen.evals(q.Mpc, [basis[i]], check_eps2=1e-4, real=True)
    g.message("Compare to: %g" % fg_feval[i])

    g.mem_report(details=False)

# now discard original basis
del fg_basis
del fg_cevec
g.message("Memory information after discarding original basis:")
g.mem_report()

# coarse grid
cgrid = params["cgrid"](q.Mpc.grid[0])
b = g.block.map(cgrid, basis)

# cheby on coarse grid
cop = params["cmatrix"](q.Mpc, b)

# implicitly restarted lanczos on coarse grid
irl = params["method_evec"]

# start vector
cstart = g.vcomplex(cgrid, nbasis)
cstart[:] = g.vcomplex([1] * nbasis, nbasis)

g.mem_report()

# basis
northo = params["northo"]
for i in range(northo):
    g.message("Orthonormalization round %d" % i)
    b.orthonormalize()

g.mem_report()

# now define coarse-grid operator
g.message(
    "Test precision of promote-project chain: %g"
    % (g.norm2(cstart - b.project * b.promote * cstart) / g.norm2(cstart))
)

g.mem_report()

try:
    cevec, cev = g.load("cevec")
except g.LoadError:
    cevec, cev = irl(cop, cstart, params["checkpointer"])
    g.save("cevec", (cevec, cev))

# smoother
smoother = params["smoother"](q.Mpc)
nsmoother = params["nsmoother"]
v_fine = g.lattice(basis[0])
v_fine_smooth = g.lattice(basis[0])
try:
    ev3 = g.load("ev3")
except g.LoadError:
    ev3 = [0.0] * len(cevec)
    for i, v in enumerate(cevec):
        v_fine @= b.promote * v
        for j in range(nsmoother):
            v_fine_smooth @= smoother * v_fine
            v_fine @= v_fine_smooth / g.norm2(v_fine_smooth) ** 0.5
        ev_smooth = g.algorithms.eigen.evals(
            q.Mpc, [v_fine], check_eps2=1e-2, real=True
        )
        ev3[i] = ev_smooth[0]
        g.message("Eigenvalue %d = %.15g" % (i, ev3[i]))
    g.save("ev3", ev3)

# tests
start = g.lattice(basis[0])
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
start *= 1.0 / g.norm2(start) ** 0.5


def save_history(fn, history):
    f = open(fn, "wt")
    for i, v in enumerate(history):
        f.write("%d %.15E\n" % (i, v))
    f.close()


test_solver = params["test_solver"]
solver = g.algorithms.inverter.sequence(
    g.algorithms.inverter.coarse_deflate(cevec, basis, ev3), test_solver
)(q.Mpc)
v_fine[:] = 0
solver(v_fine, start)
save_history("cg_test.defl_all_ev3", test_solver.history)

solver = g.algorithms.inverter.sequence(
    g.algorithms.inverter.coarse_deflate(
        cevec[0 : len(basis)], basis, ev3[0 : len(basis)]
    ),
    params["test_solver"],
)(q.Mpc)
v_fine[:] = 0
solver(v_fine, start)
save_history("cg_test.defl_full", test_solver.history)

v_fine[:] = 0
test_solver(q.Mpc)(v_fine, start)
save_history("cg_test.undefl", test_solver.history)

# save in rbc format
g.save("lanczos.output", [basis, cevec, ev3], params["format"])
