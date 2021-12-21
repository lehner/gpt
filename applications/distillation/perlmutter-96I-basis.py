#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import os, glob, sys

# configure
cnr = g.default.get("--config", None)
t_groups = g.default.get_int("--t_groups", 1)
t_group = g.default.get_int("--t_group", 0)
config = glob.glob(f"/global/cfs/projectdirs/mp13/lehner/96I/ckpoint_lat.{cnr}")
assert len(config) == 1
config = config[0]
destination = f"/pscratch/sd/l/lehner/distillation/{cnr}_basis"

t_smear_thick = 1
rho_smear = 0.1
n_smear = 30

g.default.push_verbose("irl_convergence", True)

irl = g.algorithms.eigen.irl(
    {
        "Nk": 220,
        "Nstop": 200,
        "Nm": 250,
        "resid": 1e-14,
        "betastp": 0.0,
        "maxiter": 20,
        "Nminres": 0,
    }
)

c = g.algorithms.polynomial.chebyshev({"low": 0.009, "high": 2.3, "order": 40})

# create destination directory
if not os.path.exists(destination):
    os.makedirs(destination, exist_ok=True)

# load gauge field
g.message(f"Loading {config}")
U = g.load(config)

# smear gauge links in ultra-local manner in time but heavily in space
Nt = U[0].grid.gdimensions[3]

g.message("Plaquette before", g.qcd.gauge.plaquette(U))

config_smeared = f"{destination}/smeared_lat.{cnr}"

try:
    U = g.load(config_smeared)
except g.LoadError:
    U0 = g.copy(U)
    for t in range(Nt):
        g.message("Time slice", t)
        U_temp = [g.lattice(u) for u in U]
        for u in U_temp:
            u[:] = 0
        for dt in range(-t_smear_thick, t_smear_thick + 1):
            tp = (t + Nt + dt) % Nt
            for u_dst, u_src in zip(U_temp, U0):
                u_dst[:, :, :, tp] = u_src[:, :, :, tp]
        for i in range(n_smear):
            g.message("smear", i)
            U_temp = g.qcd.gauge.smear.stout(U_temp, rho=rho_smear)
        for u_dst, u_src in zip(U, U_temp):
            u_dst[:, :, :, t] = u_src[:, :, :, t]

    # save smeared gauge field
    g.save(config_smeared, U, g.format.nersc())
    sys.exit(0)

g.message("Plaquette after", g.qcd.gauge.plaquette(U))
for u in U:
    g.message("Unitarity violation", g.norm2(u * g.adj(u) - g.identity(u)) / g.norm2(u))
    g.message(
        "SU violation",
        g.norm2(g.matrix.det(u) - g.identity(g.complex(u.grid))) / g.norm2(u),
    )

# separate time slices and define laplace operator
U3 = [g.separate(u, 3) for u in U[0:3]]

for t in range(Nt):
    if t % t_groups != t_group:
        continue

    g.message(f"Laplace basis for time-slice {t}")

    U3_t = [u[t] for u in U3]
    grid = U3_t[0].grid

    lap = g.create.smear.laplace(
        g.covariant.shift(U3_t, boundary_phases=[1.0, 1.0, 1.0, -1.0]),
        dimensions=[0, 1, 2],
    )

    def _slap(dst, src):
        dst @= -1.0 / 6.0 * lap * src

    slap = g.matrix_operator(_slap)

    start = g.vcolor(grid)
    start[:] = g.vcolor([1, 1, 1])

    if t == 0:
        g.message(
            "Power iteration spectrum test",
            g.algorithms.eigen.power_iteration(eps=1e-7, maxiter=200)(slap, start),
        )

    evec, ev = irl(c(slap), start)
    evals = g.algorithms.eigen.evals(slap, evec, check_eps2=0.1)

    g.save(f"{destination}/basis_t{t}", [evec, evals])

    del evec
