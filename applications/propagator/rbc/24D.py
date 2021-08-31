#!/usr/bin/env python3
import gpt as g

# parameters
config = g.default.get("--config", None)
evec_light = g.default.get("--evec_light", None)

# abbreviations
pc = g.qcd.fermion.preconditioner
inv = g.algorithms.inverter

# load config
U = g.load(config)

# sloppy strange quark
strange_sloppy = g.qcd.fermion.zmobius(
    g.convert(U, g.single),
    {
        "mass": 0.0850,
        "M5": 1.8,
        "b": 1.0,
        "c": 0.0,
        "omega": [
            1.0903256131299373,
            0.9570283702230611,
            0.7048886040934104,
            0.48979921782791747,
            0.328608311201356,
            0.21664245377015995,
            0.14121112711957107,
            0.0907785101745156,
            0.05608303440064219 - 0.007537158177840385j,
            0.05608303440064219 + 0.007537158177840385j,
            0.0365221637144842 - 0.03343945161367745j,
            0.0365221637144842 + 0.03343945161367745j,
        ],
        "boundary_phases": [1.0, 1.0, 1.0, -1.0],
    },
)

# sloppy light quark
light_sloppy = g.qcd.fermion.zmobius(
    g.convert(U, g.single),
    {
        "mass": 0.00107,
        "M5": 1.8,
        "b": 1.0,
        "c": 0.0,
        "omega": [
            1.0903256131299373,
            0.9570283702230611,
            0.7048886040934104,
            0.48979921782791747,
            0.328608311201356,
            0.21664245377015995,
            0.14121112711957107,
            0.0907785101745156,
            0.05608303440064219 - 0.007537158177840385j,
            0.05608303440064219 + 0.007537158177840385j,
            0.0365221637144842 - 0.03343945161367745j,
            0.0365221637144842 + 0.03343945161367745j,
        ],
        "boundary_phases": [1.0, 1.0, 1.0, -1.0],
    },
)

# load evecs
if evec_light is not None:
    basis, cevec, ev = g.load(evec_light, grids=light_sloppy.F_grid_eo)
    cdefl = inv.coarse_deflate(cevec, basis, ev)


# strange exact
strange_exact = g.qcd.fermion.mobius(
    U,
    {
        "mass": 0.0850,
        "M5": 1.8,
        "b": 2.5,
        "c": 1.5,
        "Ls": 24,
        "boundary_phases": [1.0, 1.0, 1.0, -1.0],
    },
)

# create point source
src = g.vspincolor(U[0].grid)
rng = g.random("test")
rng.cnormal(src)
src /= g.norm2(src) ** 0.5

# solver
g.default.push_verbose("cg", True)
g.default.push_verbose("cg_convergence", True)

g.default.push_verbose("defect_correcting", True)
g.default.push_verbose("defect_correcting_convergence", True)

strange_sloppy_solver = inv.preconditioned(
    pc.eo2_kappa_ne(), inv.cg({"eps": 1e-7, "maxiter": 400})
)

# Alternative: split-grid inverter
# strange_sloppy_solver = inv.preconditioned(
#     pc.eo2_kappa_ne(), inv.split(inv.cg({"eps": 1e-7, "maxiter": 400}), mpi_split=g.default.get_ivec("--mpi_split", None, 4))
# )

pauli_villars_solver = inv.preconditioned(
    pc.eo2_ne(), inv.cg({"eps": 1e-7, "maxiter": 150})
)

strange_exact_solver = inv.defect_correcting(
    inv.mixed_precision(
        pc.mixed_dwf(strange_sloppy_solver, pauli_villars_solver, strange_sloppy),
        g.single,
        g.double,
    ),
    eps=1e-8,
    maxiter=10,
)

strange_sloppy_propagator = strange_sloppy.propagator(strange_sloppy_solver)
strange_exact_propagator = strange_exact.propagator(strange_exact_solver)

# create propagator
g.message("Create sloppy strange propagator")
dst_sloppy = g.convert(strange_sloppy_propagator * g.convert(src, g.single), g.double)

g.message("Create exact strange propagator")
dst_exact = g(strange_exact_propagator * src)

eps2 = g.norm2(dst_sloppy - dst_exact) / g.norm2(dst_sloppy)
g.message(f"Relative difference: {eps2**0.5}")
