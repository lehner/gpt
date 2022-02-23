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

rat = g.algorithms.rational

np = 12
zol = g.algorithms.rational.zolotarev_inverse_square_root(0.1, 4.5, np)
g.message(zol)

r = rat.rational_function(zol.zeros, zol.poles, zol.norm, mscg)
r_inv = r.inv()

g.message(r)
for x in numpy.arange(0.1, 4.5, 0.05):
    num = numpy.prod(x * x - zol.zeros) * zol.norm
    den = numpy.prod(x * x - zol.poles)
    assert abs(r(x * x) - num / den) < 1e-12
    assert abs(r_inv(x * x) - den / num) < 1e-12

# we test arbitrary function
zeros = numpy.array([0.3, 0.5])
poles = numpy.array([0.1, 0.4, 0.9])
rp = rat.rational_function(zeros, poles)
g.message(rp)
for y in numpy.arange(1.0, 4.5, 0.05):
    num = numpy.prod(y - zeros)
    den = numpy.prod(y - poles)
    assert abs(rp(y) - num / den) < 1e-12

rr = r(mat)

# create point source
src = rng.cnormal(g.vspincolor(w.F_grid_eo))

psi, phi = g.lattice(src), g.lattice(src)
psi @= rr * src

inv = g.algorithms.inverter
cg = inv.cg({"eps": 1e-12, "maxiter": 1024})
g.default.set_verbose("cg")


def mat_shift(dst, src, s):
    dst @= mat * src - s * src


phi @= src
for i in range(np):
    g.message(f" (mat - {zol.zeros[i]})/(mat - {zol.poles[i]})")
    mat_inv = cg(lambda dst, src: mat_shift(dst, src, zol.poles[i]))
    tmp = mat_inv(phi)
    mat_shift(phi, tmp, zol.zeros[i])
phi *= zol.norm

g.message("Testing rat_polynomial against exact calculation")
eps = g.inner_product(src, phi).real - g.inner_product(src, psi).real
g.message(f"  = {abs(eps):e}")
assert abs(eps) < 1e-8

# test fundamental definition
eps2 = g.norm2(mat * rr * rr * src - src) / g.norm2(src)
g.message(f"Test 1/sqrt(mat) approximation: {eps2}")
assert eps2 < 1e-15
