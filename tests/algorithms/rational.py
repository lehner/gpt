#!/usr/bin/env python3
#
# Authors: Mattia Bruno 2022
#          Raphael Lehner 2022
#          Christoph Lehner 2022
#
import numpy
import gpt as g

# load configuration
precision = g.double
rng = g.random("rational")
U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], precision), rng)

# use the gauge configuration grid
grid = U[0].grid

# quark
w = g.qcd.fermion.wilson_clover(
    U,
    {
        "kappa": 0.13565,
        "csw_r": 2.0171 / 2.0,  # for now test with very heavy quark
        "csw_t": 2.0171 / 2.0,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)

eo2_odd = g.qcd.fermion.preconditioner.eo2_ne(parity=g.odd)
mat = eo2_odd(w).Mpc

# short-cuts
inv = g.algorithms.inverter
rat = g.algorithms.rational

# inverters
g.default.set_verbose("multi_shift_cg")
mscg = inv.multi_shift_cg({"eps": 1e-8, "maxiter": 1024})

g.default.set_verbose("cg")
cg = inv.cg({"eps": 1e-12, "maxiter": 1024})

g.default.set_verbose("multi_shift_fom")
msfom = inv.multi_shift_fom({"restartlen": 10, "maxiter": 1024})

g.default.set_verbose("multi_shift_fgmres")
msfgmres = inv.multi_shift_fgmres({"restartlen": 10, "maxiter": 1024})

# create point source
src = rng.cnormal(g.vspincolor(w.F_grid_eo))

psi, phi = g.lattice(src), g.lattice(src)


def mat_shift(dst, src, s):
    dst @= mat * src - s * src

# arnoldi for deflation
g.default.set_verbose("arnoldi")
a = g.algorithms.eigen.arnoldi(
    Nmin=200, Nmax=300, Nstep=50, Nstop=20, resid=1e-5
)

mms = g.copy(src)


def msq(dst, src):
    mat(mms, src)
    mat(dst, mms)

# eigenvalues and eigenvectors for deflation
evec, evals = a(msq, src)
idx = evals.argsort()
evals = evals[idx]
evec = [evec[i] for i in idx]


# test arbitrary function

zeros = numpy.array([0.3, 0.5])
poles = numpy.array([0.1, 0.4, 0.9])
rp = rat.rational_function(zeros, poles)
g.message(rp)

for y in numpy.arange(1.0, 4.5, 0.05):
    num = numpy.prod(y - zeros)
    den = numpy.prod(y - poles)
    assert abs(rp(y) - num / den) < 1e-12


# test zolotarev_inverse_square_root

# number of poles
npz = 12

zol = rat.zolotarev_inverse_square_root(0.1, 4.5, npz)
g.message(zol)

rz = rat.rational_function(zol.zeros, zol.poles, zol.norm, mscg)
rz_inv = rz.inv()
g.message(rz)

for x in numpy.arange(0.1, 4.5, 0.05):
    num = numpy.prod(x * x - zol.zeros) * zol.norm
    den = numpy.prod(x * x - zol.poles)
    assert abs(rz(x * x) - num / den) < 1e-12
    assert abs(rz_inv(x * x) - den / num) < 1e-12

rrz = rz(mat)
psi @= rrz * src
phi @= src

for i in range(npz):
    g.message(f" (mat - {zol.zeros[i]})/(mat - {zol.poles[i]})")
    mat_inv = cg(lambda dst, src: mat_shift(dst, src, zol.poles[i]))
    tmp = mat_inv(phi)
    mat_shift(phi, tmp, zol.zeros[i])
phi *= zol.norm

g.message("Testing zolotarev_rational_polynomial against exact calculation")
eps = g.inner_product(src, phi).real - g.inner_product(src, psi).real
g.message(f"  = {abs(eps):e}")
assert abs(eps) < 1e-8

# test fundamental definition
eps2 = g.norm2(mat * rrz * rrz * src - src) / g.norm2(src)
g.message(f"Test zolotarev 1/sqrt(mat) approximation: {eps2}")
assert eps2 < 1e-12

# test zolotarev_sign and msfom with LR deflation
signz = rat.zolotarev_sign(
    {"eps": 1e-5, "inverter": msfom, "low": 0.4, "high": 3.4}
)
ssignz = signz(mat, msq_evals=evals[0:10], msq_evec=evec[0:10])
eps2 = g.norm2(ssignz * ssignz * src - src) / g.norm2(src)
g.message(f"Test zolotarev sign(mat) approximation: {eps2}")
assert eps2 < 1e-5

# test neuberger_inverse_square_root

# number of poles
npn = 24

neu = rat.neuberger_inverse_square_root(2.3, 2.2, npn)
g.message(neu)

rn = rat.rational_function(neu.zeros, neu.poles, neu.norm, mscg)
g.message(rn)

for x in numpy.arange(1.1, 3.5, 0.05):
    num = numpy.prod(x * x - neu.zeros) * neu.norm
    den = numpy.prod(x * x - neu.poles)
    assert abs(rn(x * x) - num / den) < 1e-12

rrn = rn(mat)
psi @= rrn * src
phi @= src

for i in range(npn - 1):
    g.message(f" (mat - {neu.zeros[i]})/(mat - {neu.poles[i]})")
    mat_inv = cg(lambda dst, src: mat_shift(dst, src, neu.poles[i]))
    tmp = mat_inv(phi)
    mat_shift(phi, tmp, neu.zeros[i])
g.message(f" 1/(mat - {neu.poles[-1]})")
mat_inv = cg(lambda dst, src: mat_shift(dst, src, neu.poles[-1]))
tmp = mat_inv(phi)
phi @= tmp * neu.norm

g.message("Testing neuberger_rational_polynomial against exact calculation")
eps = g.inner_product(src, phi) - g.inner_product(src, psi)
g.message(f"  = {abs(eps):e}")
assert abs(eps) < 1e-8

# test fundamental definition
eps2 = g.norm2(mat * rrn * rrn * src - src) / g.norm2(src)
g.message(f"Test neuberger 1/sqrt(mat) approximation: {eps2}")
assert eps2 < 1e-12

# test neuberger_sign and msfgmres with LR deflation
signn = rat.neuberger_sign(
    {"eps": 1e-5, "inverter": msfgmres, "m": 1.9, "r": 1.5}
)
ssignn = signn(mat, msq_evals=evals[0:10], msq_evec=evec[0:10])
eps2 = g.norm2(ssignn * ssignn * src - src) / g.norm2(src)
g.message(f"Test neuberger sign(mat) approximation: {eps2}")
assert eps2 < 1e-5
