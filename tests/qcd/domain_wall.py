#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys
import time


# load configuration
# U = g.load("/hpcgpfs01/work/clehner/configs/16I_0p01_0p04/ckpoint_lat.IEEE64BIG.1100")
rng = g.random("test")
U = g.qcd.gauge.random(g.grid([8, 8, 8, 8], g.double), rng, scale=2.0)
g.message("Plaquette:", g.qcd.gauge.plaquette(U))

# do everything in single-precision
U = g.convert(U, g.single)

# use the gauge configuration grid
grid = U[0].grid

# mobius <> zmobius domain wall quark
mobius_params = {
    "mass": 0.08,
    "M5": 1.8,
    "b": 1.5,
    "c": 0.5,
    "Ls": 12,
    "boundary_phases": [1.0, 1.0, 1.0, 1.0],
}

qm = g.qcd.fermion.mobius(g.qcd.gauge.unit(grid), mobius_params)


# test operator update
start = g.vspincolor(qm.F_grid)
rng.cnormal(start)
qm_new = g.qcd.fermion.mobius(U, mobius_params)
qm.update(U)
eps2 = g.norm2(qm * start - qm_new * start) / g.norm2(start)
g.message(f"Operator update test: {eps2}")
assert eps2 < 1e-15


# study kernel spectrum
w = g.qcd.fermion.wilson_clover(
    U,
    {
        "mass": -qm.params["M5"],
        "csw_r": 0,
        "csw_t": 0,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)

# solver
inv = g.algorithms.inverter
cg = inv.cg({"eps": 1e-4, "maxiter": 1000})


def H5_denom(dst, src):
    dst @= g.gamma[5] * (2 * src + (qm.params["b"] - qm.params["c"]) * w * src)


inv_H5_denom = cg(H5_denom)


def H5(dst, src):
    # g5 * Dkernel
    # Dkernel = (b+c)*w / (2 + (b-c)*w)
    dst @= inv_H5_denom * w * src
    dst *= qm.params["b"] + qm.params["c"]


# arnoldi to get an idea of entire spectral range of w
start = g.vspincolor(w.F_grid)
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
g.default.push_verbose("arnoldi", False)
a = g.algorithms.eigen.arnoldi(Nmin=20, Nmax=20, Nstep=0, Nstop=20, resid=1)
_, evals_H5 = a(H5, start)
g.default.pop_verbose()
g.message(evals_H5)
# H5 spectrum for 16c RBC lattice (b+c=2): [-2.72214117, ..., 2.68753147]
# H5 spectrum for random lattice:  [-2.69115638, ..., 2.69136854]

qz = g.qcd.fermion.zmobius(
    U,
    {
        "mass": 0.08,
        "M5": 1.8,
        "b": 1.0,
        "c": 0.0,
        "omega": [
            0.17661651536320583 + 1j * (0.14907774771612217),
            0.23027432016909377 + 1j * (-0.03530801572584271),
            0.3368765581549033 + 1j * (0),
            0.7305711010541054 + 1j * (0),
            1.1686138337986505 + 1j * (0.3506492418109086),
            1.1686138337986505 + 1j * (-0.3506492418109086),
            0.994175013717952 + 1j * (0),
            0.5029903152251229 + 1j * (0),
            0.23027432016909377 + 1j * (0.03530801572584271),
            0.17661651536320583 + 1j * (-0.14907774771612217),
        ],
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)


# create point source
src = g.mspincolor(grid)
g.create.point(src, [0, 1, 0, 0])

# solver
pc = g.qcd.fermion.preconditioner
inv = g.algorithms.inverter
cg = inv.cg({"eps": 1e-5, "maxiter": 1000})
cg_kappa = inv.cg({"eps": 1e-5, "maxiter": 1000})
cg_e = inv.cg({"eps": 1e-8, "maxiter": 1000})

slv_5d = inv.preconditioned(pc.eo2_ne(), cg)

# kappa: RBC/UKQCD solver for zmobius strange quark
# use cg_kappa instead of identical cg to keep
# track of iteration counts separately
slv_5d_kappa = inv.preconditioned(pc.eo2_kappa_ne(), cg_kappa)
slv_5d_e = inv.preconditioned(pc.eo2_ne(), cg_e)

# To calculate Jq5 which is necessary for the residual mass, we need a solver for the bulk propgator
slv_qm_bulk = qm.bulk_propagator(slv_5d)
slv_qm_e = qm.propagator(slv_5d_e)
slv_qz = qz.propagator(slv_5d)
slv_qz_kappa = qz.propagator(slv_5d_kappa)
slv_madwf = qm.propagator(pc.mixed_dwf(slv_5d, slv_5d, qz))
slv_madwf_dc = qm.propagator(
    inv.defect_correcting(pc.mixed_dwf(slv_5d, slv_5d, qz), eps=1e-6, maxiter=10)
)


# inverse one spin color src
src_sc = rng.cnormal(g.vspincolor(grid))
dst_dwf_sc = g(slv_qm_e * src_sc)

# test madwf
dst_madwf_sc, dst_madwf_sc2 = g(slv_madwf * [src_sc, src_sc])
eps2 = g.norm2(dst_madwf_sc - dst_dwf_sc) / g.norm2(dst_dwf_sc)
g.message(f"MADWF test: {eps2}")
assert eps2 < 5e-4

eps2 = g.norm2(dst_madwf_sc - dst_madwf_sc2) / g.norm2(dst_madwf_sc)
g.message(f"MADWF multi-rhs test: {eps2}")
assert eps2 < 1e-13

# test madwf with defect_correcting
dst_madwf_dc_sc = g(slv_madwf_dc * src_sc)
eps2 = g.norm2(dst_madwf_dc_sc - dst_dwf_sc) / g.norm2(dst_dwf_sc)
g.message(f"MADWF defect_correcting test: {eps2}")
assert eps2 < 1e-10


# propagator
dst_qm = g.mspincolor(grid)
dst_qz = g.mspincolor(grid)
dst_qz_kappa = g.mspincolor(grid)

# Solve for the 5d and 4d propagator
# qm.bulk_propagator_to_propagator * qm.bulk_propagator(slv) == qm.propagator(slv)
dst_qm_bulk = g(slv_qm_bulk.grouped(3) * src)
dst_qm @= qm.bulk_propagator_to_propagator * dst_qm_bulk

dst_qz @= slv_qz * src
dst_qz_kappa @= slv_qz_kappa * src

# test similarity transformated solve
eps2 = g.norm2(dst_qz - dst_qz_kappa) / g.norm2(dst_qz)
g.message(f"Kappa similarity transformed solve: {eps2}")
assert eps2 < 1e-6
assert len(cg.history) > len(cg_kappa.history)

# calculate J5q
p = qm.J5q(dst_qm_bulk)
J5q = g.slice(g.trace(p * g.adj(p)), 3)

J5q_ref = [
    1.3814802514389157e-05,
    8.530755621904973e-06,
    7.805140739947092e-06,
    5.135065748618217e-06,
    5.082366897113388e-06,
    4.842216185352299e-06,
    7.341488071688218e-06,
    7.706037649768405e-06,
]

eps = np.linalg.norm(np.array(J5q) - np.array(J5q_ref))
g.message(f"J5q test: {eps}")
assert eps < 1e-5

# compute conserved current divergence
div = g.mspin(grid)
div[:] = 0

for mu in range(4):
    tmp = qm.conserved_vector_current(dst_qm_bulk, src, dst_qm_bulk, src, mu)
    tmp -= g.cshift(tmp, mu, -1)
    div += g.color_trace(tmp)

div = g(g.trace(g.adj(div) * div))

g.message("div(conserved_current) contact term", div[0, 1, 0, 0].real)

div[0, 1, 0, 0] = 0

eps = g.sum(div).real
g.message(f"div(conserved_current) = {eps} without contact term")
assert eps < 1e-11

# compute partially conserved axial current divergence (zero momentum projected)
AP = g.slice(
    g.trace(qm.conserved_axial_current(dst_qm_bulk, src, dst_qm_bulk, src, 3) * g.gamma[5]), 3
)
PP = g.slice(g.trace(dst_qm * g.adj(dst_qm)), 3)

Nt = grid.gdimensions[3]
for t in range(Nt):
    dAP_t = AP[t] - AP[(t - 1 + Nt) % Nt]
    mass_term = (PP[t] * 0.08 + J5q[t]) * 2.0
    eps = abs(dAP_t - mass_term) / abs(dAP_t + mass_term)
    if t != 0:
        g.message(f"axial vector current divergence residuum at t={t}: {eps}")
        assert eps < 1e-5

# two-point
# correlator_ref= g.slice(g.trace(dst_qm_e * g.adj(dst_qm_e)), 3)
correlator_ref = [
    0.5534145832061768,
    0.2355920523405075,
    0.08622127771377563,
    0.05764763802289963,
    0.05238068848848343,
    0.057377591729164124,
    0.08141942322254181,
    0.21931196749210358,
]
correlator_qm = g.slice(
    g.trace(g.gamma[0] * g.gamma[0] * dst_qm * g.gamma[0] * g.gamma[0] * g.adj(dst_qm)),
    3,
)
correlator_qz = g.slice(g.trace(dst_qz * g.adj(dst_qz)), 3)

# output
eps_qm = 0.0
eps_qz = 0.0
for t in range(len(correlator_ref)):
    eps_qm += (correlator_qm[t].real - correlator_ref[t]) ** 2.0
    eps_qz += (correlator_qz[t].real - correlator_ref[t]) ** 2.0
    g.message(t, correlator_qm[t].real, correlator_qz[t].real, correlator_ref[t])
eps_qm = eps_qm**0.5 / len(correlator_ref)
eps_qz = eps_qz**0.5 / len(correlator_ref)
g.message("Test results: %g %g" % (eps_qm, eps_qz))
assert eps_qm < 1e-5
assert eps_qz < 5e-4

# test G(m1) - G(m2) = (m2 - m1) * G(m1) * G(m2)
m1 = 0.11
m2 = 0.24
qm1 = g.qcd.fermion.mobius(U, mass=m1, M5=1.8, b=1.5, c=0.5, Ls=6, boundary_phases=[1, 1, 1, -1])
qm2 = g.qcd.fermion.mobius(U, mass=m2, M5=1.8, b=1.5, c=0.5, Ls=6, boundary_phases=[1, 1, 1, -1])
G1 = qm1.propagator(slv_5d_e)
G2 = qm2.propagator(slv_5d_e)

eps2 = g.norm2((m2 - m1) * G1 * G2 * src_sc - (G1 * src_sc - G2 * src_sc)) / g.norm2(src_sc)
g.message(f"Test vector mass behavior: {eps2}")
assert eps2 < 1e-13


# now test axial mass behavior; first test action at double precision
m0 = 0.35
m1 = 0.11
m2 = 0.24

U = g.qcd.gauge.random(g.grid([8, 8, 8, 8], g.double), rng, scale=2.0)
qm1 = g.qcd.fermion.mobius(
    U,
    mass_plus=m0,
    mass_minus=m1,
    M5=1.8,
    b=1.5,
    c=0.5,
    Ls=12,
    boundary_phases=[1, 1, 1, -1],
)


# reference implementation
def D_DWF(dst, src, b, c, mass_plus, mass_minus):
    D_W = g.qcd.fermion.wilson_clover(
        U,
        mass=-1.8,
        csw_r=0.0,
        csw_t=0.0,
        nu=1.0,
        xi_0=1.0,
        isAnisotropic=False,
        boundary_phases=[1, 1, 1, -1],
    )

    src_s = g.separate(src, 0)
    dst_s = [g.lattice(s) for s in src_s]

    Ls = len(src_s)

    src_plus_s = []
    src_minus_s = []
    for s in range(Ls):
        src_plus_s.append(g(0.5 * src_s[s] + 0.5 * g.gamma[5] * src_s[s]))
        src_minus_s.append(g(0.5 * src_s[s] - 0.5 * g.gamma[5] * src_s[s]))
    for d in dst_s:
        d[:] = 0
    for s in range(Ls):
        dst_s[s] += b * D_W * src_s[s] + src_s[s]
    for s in range(1, Ls):
        dst_s[s] += c * D_W * src_plus_s[s - 1] - src_plus_s[s - 1]
    for s in range(0, Ls - 1):
        dst_s[s] += c * D_W * src_minus_s[s + 1] - src_minus_s[s + 1]
    dst_s[0] -= mass_plus * (c * D_W * src_plus_s[Ls - 1] - src_plus_s[Ls - 1])
    dst_s[Ls - 1] -= mass_minus * (c * D_W * src_minus_s[0] - src_minus_s[0])
    dst @= g.merge(dst_s, 0)


vsrc = rng.cnormal(g.vspincolor(qm1.F_grid))
vdst = g(qm1 * vsrc)
vdst2 = g.lattice(vdst)
D_DWF(vdst2, vsrc, 1.5, 0.5, m0, m1)
eps2 = g.norm2(vdst2 - vdst) / g.norm2(vdst)
g.message(f"Test DWF implementation with separate left and right mass: {eps2}")
assert eps2 < 1e-28


# remaining tests again with single-precision for speed (did run once in double-precision)
U = g.convert(U, g.single)

qm1 = g.qcd.fermion.mobius(
    U,
    mass_plus=m0,
    mass_minus=m1,
    M5=1.8,
    b=1.5,
    c=0.5,
    Ls=6,
    boundary_phases=[1, 1, 1, -1],
)
qm2 = g.qcd.fermion.mobius(
    U,
    mass_plus=m0,
    mass_minus=m2,
    M5=1.8,
    b=1.5,
    c=0.5,
    Ls=6,
    boundary_phases=[1, 1, 1, -1],
)

G1 = qm1.propagator(slv_5d_e)
G2 = qm2.propagator(slv_5d_e)

Pplus = (g.gamma["I"].tensor() + g.gamma[5].tensor()) * 0.5
Pminus = (g.gamma["I"].tensor() - g.gamma[5].tensor()) * 0.5

# test left-handed axial behavior:
# G(m,m1) = (G0 + m Pp + m1 Pm)^-1
# ->
# G(m,m1)^-1 G(m,m2) = (G0 + m Pp + m1 Pm) (G0 + m Pp + m2 Pm)^-1
#                    = (G0 + m Pp + m2 Pm) (G0 + m Pp + m2 Pm)^-1 + (m1-m2) Pm (G0 + m Pp + m2 Pm)^-1
#                    = 1 + (m1-m2) Pm (G0 + m Pp + m2 Pm)^-1
# ->
# G(m,m2) - G(m,m1) = (m1 - m2) G(m,m1) Pm G(m,m2)
# or
# G(m,m1) - G(m,m2) = (m2 - m1) G(m,m2) Pm G(m,m1)
eps2 = g.norm2((m2 - m1) * G2 * Pminus * G1 * src_sc - (G1 * src_sc - G2 * src_sc)) / g.norm2(
    src_sc
)
g.message(f"Test axial mass behavior: {eps2}")
assert eps2 < 1e-13

# G(m1,m) - G(m2,m) = (m2 - m1) G(m2,m) Pp G(m1,m)
qm1 = g.qcd.fermion.mobius(
    U,
    mass_plus=m1,
    mass_minus=m0,
    M5=1.8,
    b=1.5,
    c=0.5,
    Ls=6,
    boundary_phases=[1, 1, 1, -1],
)
qm2 = g.qcd.fermion.mobius(
    U,
    mass_plus=m2,
    mass_minus=m0,
    M5=1.8,
    b=1.5,
    c=0.5,
    Ls=6,
    boundary_phases=[1, 1, 1, -1],
)

G1 = qm1.propagator(slv_5d_e)
G2 = qm2.propagator(slv_5d_e)

eps2 = g.norm2((m2 - m1) * G2 * Pplus * G1 * src_sc - (G1 * src_sc - G2 * src_sc)) / g.norm2(src_sc)
g.message(f"Test axial mass behavior: {eps2}")
assert eps2 < 1e-13
