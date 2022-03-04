#!/usr/bin/env python3
#
# Authors: Mattia Bruno 2022
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

# inverters
inv = g.algorithms.inverter

g.default.set_verbose("multi_shift_cg")
mscg = inv.multi_shift_cg({"eps": 1e-8, "maxiter": 1024})

g.default.set_verbose("cg")
cg = inv.cg({"eps": 1e-12, "maxiter": 1024})

rat = g.algorithms.rational

# create point source
src = rng.cnormal(g.vspincolor(w.F_grid_eo))


def test_matrix_application(rp):
    rp_src = g(rp(mat) * src)
    rp_src /= rp.norm

    for p in rp.poles:
        g.message(p)
        rp_src @= mat * rp_src - p * rp_src

    dst = g.copy(src)
    for z in rp.zeros:
        g.message(z)
        dst @= mat * dst - z * dst

    eps2 = g.norm2(dst - rp_src) / g.norm2(dst)
    g.message(f"Test of matrix application: {eps2}")
    assert eps2 < 1e-9


# test function with more poles than zeros
zeros = numpy.array([0.32, 0.551])
poles = numpy.array([0.17, 0.42, 0.911, 0.21])
norm = 2.39
rp = rat.rational_function(zeros, poles, norm)
g.message(rp)

for y in numpy.arange(1.0, 4.5, 0.05):
    num = numpy.prod(y - zeros)
    den = numpy.prod(y - poles)
    assert abs(rp(y) - num / den * norm) < 1e-12

test_matrix_application(rat.rational_function(zeros, -poles, norm, mscg))

# test function with same number of poles and zeros
zeros = numpy.array([0.32, 0.551, 0.1234])
poles = numpy.array([0.17, 0.42, 0.911])
rp = rat.rational_function(zeros, poles, norm)
rp_inv = rp.inv()
g.message(rp)

for y in numpy.arange(1.0, 4.5, 0.05):
    num = numpy.prod(y - zeros)
    den = numpy.prod(y - poles)
    assert abs(rp(y) - num / den * norm) < 1e-12
    assert abs(rp_inv(y) - den / num / norm) < 1e-12

test_matrix_application(rat.rational_function(zeros, -poles, norm, mscg))

# test zolotarev_inverse_square_root

zol = rat.zolotarev_inverse_square_root(0.1, 4.5, 12)
g.message(zol)

rz = rat.rational_function(zol.zeros, zol.poles, zol.norm, mscg)
g.message(rz)

eps = max([abs(rz(x * x) * x - 1.0) for x in numpy.arange(0.1, 4.5, 0.05)])
g.message(f"Maximal tested error of Zolotarev: {eps}")

rrz = rz(mat)

# test fundamental definition
eps2 = g.norm2(mat * rrz * rrz * src - src) / g.norm2(src)
g.message(f"Test zolotarev 1/sqrt(mat) approximation: {eps2}")
assert eps2 < 1e-12


# test neuberger_inverse_square_root

neu = rat.neuberger_inverse_square_root(0.1, 4.5, 24)
g.message(neu)

rn = rat.rational_function(neu.zeros, neu.poles, neu.norm, mscg)
g.message(rn)

eps = max([abs(rn(x * x) * x - 1.0) for x in numpy.arange(0.1, 4.5, 0.05)])
g.message(f"Maximal tested error of Neuberger: {eps}")

rrn = rn(mat)

# test fundamental definition
eps2 = g.norm2(mat * rrn * rrn * src - src) / g.norm2(src)
g.message(f"Test neuberger 1/sqrt(mat) approximation: {eps2}")
assert eps2 < 1e-12
