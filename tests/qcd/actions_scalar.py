#!/usr/bin/env python3
#
# Authors: Mattia Bruno 2020
#          Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#

import gpt as g

g.default.set_verbose("hmc")
grid = g.grid([8, 8, 8, 8], g.double)

rng = g.random("action_scalar")

phi = g.complex(grid)
rng.normal(phi, sigma=0.1)
phi[:].imag = 0

mass = 0.25
lam = 0.1234
act = g.qcd.actions.scalar.phi4

dphi = g.complex(grid)
rng.normal(dphi)
dphi[:].imag = 0

phip = g.complex(grid)
phim = g.complex(grid)

eps = 1e-4

phip @= phi + dphi * eps
phim @= phi - dphi * eps
ap = act(phip, mass, lam)
am = act(phim, mass, lam)
da = 2 / 3 * (ap() - am())

phip @= phi + dphi * eps * 2.0
phim @= phi - dphi * eps * 2.0
ap = act(phip, mass, lam)
am = act(phim, mass, lam)
da += 1 / 12 * (-ap() + am())

da *= 1 / eps

frc = g.lattice(phi)
aref = act(phi, mass, lam)
aref.setup_force()

frc @= aref.force(phi)
daa = g.inner_product(frc, dphi).real

dd = abs(da / daa - 1.0)
g.message(f"Relative difference numerical and analytic force = {dd:g}")
assert dd < 1e-10

#######

rho = g.lattice(phi)
rng.normal(rho)
rho[:].imag = 0

gc = 0.1234
act = g.qcd.actions.scalar.rho_phi2

eps = 1e-4

phip @= phi + dphi * eps
phim @= phi - dphi * eps
ap = act(phip, rho, gc)
am = act(phim, rho, gc)
da = 2 / 3 * (ap() - am())

phip @= phi + dphi * eps * 2.0
phim @= phi - dphi * eps * 2.0
ap = act(phip, rho, gc)
am = act(phim, rho, gc)
da += 1 / 12 * (-ap() + am())

da *= 1 / eps

frc = g.lattice(phi)
aref = act(phi, rho, gc)
aref.setup_force()

frc @= aref.force(phi)
daa = g.inner_product(frc, dphi).real

dd = abs(da / daa - 1.0)
g.message(f"Relative difference numerical and analytic force = {dd:g}")
assert dd < 1e-10

phip @= phi + dphi * eps
phim @= phi - dphi * eps
ap = act(rho, phip, gc)
am = act(rho, phim, gc)
da = 2 / 3 * (ap() - am())

phip @= phi + dphi * eps * 2.0
phim @= phi - dphi * eps * 2.0
ap = act(rho, phip, gc)
am = act(rho, phim, gc)
da += 1 / 12 * (-ap() + am())

da *= 1 / eps

frc = g.lattice(phi)
aref = act(rho, phi, gc)
aref.setup_force()

frc @= aref.force(phi)
daa = g.inner_product(frc, dphi).real

dd = abs(da / daa - 1.0)
g.message(f"Relative difference numerical and analytic force = {dd:g}")
assert dd < 1e-10
