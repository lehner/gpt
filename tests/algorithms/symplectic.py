#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno 2021
#
import gpt as g

# load configuration
rng = g.random("test")
grid = g.grid([4, 4, 4, 8], g.double)

q = g.real(grid)
p = g.group.cartesian(q)
p0 = g.lattice(p)

# harmonic oscillator

# p^2 / 2m , m=1
a0 = g.qcd.scalar.action.mass_term()

# q^2 * k / 2
k = 0.1234
a1 = g.qcd.scalar.action.mass_term(k)

# starting config
q[:] = 0
rng.element(p)
p0 @= p

# evolution
tau = 1.0

# integrators
sympl = g.algorithms.integrator.symplectic

ip = sympl.update_p(p, lambda : a1.gradient(q,q))
iq = sympl.update_q(q, lambda : a0.gradient(p,p))

# ref solution obtained with Euler scheme
M = 1000
eps = tau / M
for k in range(M):
    ip(eps)
    iq(eps)
qref = g.lattice(q)
qref @= q

integrator = [sympl.leap_frog, sympl.OMF2, sympl.OMF4]
criterion = [1e-5, 1e-8, 1e-12]

for i in range(3):
    # initial config
    q[:] = 0
    p @= p0
    
    # solve
    integrator[i](10, ip, iq)(tau)
    
    eps = g.norm2(q - qref)
    print(f'{integrator[i].__name__ : <10}: |q - qref|^2 = {eps:.4e}')
    assert eps < criterion[i]
