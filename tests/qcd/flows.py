#!/usr/bin/env python3
#
# Flow-type models
#

import gpt as g

# general setup
rng = g.random("test")
U = g.qcd.gauge.random(g.grid([4,4,4,4], g.double), rng)
rad = g.ad.reverse

# specific even/odd pattern
even, odd = g.even_odd_projectors(U[0].grid)
full = g(even + odd)
none = g(0 * full)

# plaquette-type flow (P-FTHMC with local weights)
rho = g.complex(even.grid)
rho[:] = 0.12
params = [rho]

description = [
    (
    [(rho, g.path().f(nu).f(mu).b(nu).b(mu)) for nu in range(4) if mu != nu]
    +
    [(rho, g.path().b(nu).f(mu).f(nu).b(mu)) for nu in range(4) if mu != nu]
    )
    for mu in range(4)
]

pt_e = [
    g.qcd.gauge.smear.directional_parallel_transport(
        U,
        description[j],
        j,
        full,
        even,
        params
    )
    for j in range(4)
]

pt_o = [
    g.qcd.gauge.smear.directional_parallel_transport(
        U,
        description[j],
        j,
        full,
        odd,
        params
    )
    for j in range(4)
]


#L = np.array(even.grid.gdimensions)
#coor = g.coordinates(even)
#coor = (coor + L//2) % L - L//2
#r2 = np.sum(coor*coor, axis=1)
#rho[:] = 0.12*np.exp(-r2)
#rho = rad.node(rho)
#rho = 0.12

# first establish agreement with specialized local_stout
Uprime0 = pt_e[1](U + params)[0:4]
Uprime1 = g.qcd.gauge.smear.local_stout(rho=0.12, dimension=1, checkerboard=g.even)(U)
eps2 = sum(g.norm2(x - y) / g.norm2(x) for x, y in zip(Uprime0, Uprime1))
g.message(f"Test agreement with P-FTHMC: {eps2}")
assert eps2 < 1e-28

# next test that it works in the Jacobian
# version with separate parameters for the layers
a1 = g.qcd.gauge.action.iwasaki(6)
a1 = a1.transformed(pt_o[0], indices=[0,1,2,3,4], projection=[0,1,2,3])
a1 = a1.transformed(pt_e[0], indices=[0,1,2,3,5], projection=[0,1,2,3,4])
a1 = a1.transformed(pt_o[1], indices=[0,1,2,3,6], projection=[0,1,2,3,4,5])
rho1=g.copy(rho)
rho2=g.copy(rho)
rho3=g.copy(rho)
params2=[rho1,rho2,rho3]

a1.assert_gradient_error(rng, U + params2, U + params2, 1e-4, 1e-7)


# version with separate parameters for the layers
a1 = g.qcd.gauge.action.iwasaki(6)
a1 = a1.transformed(pt_o[0], indices=[0,1,2,3,4], projection=[0,1,2,3])
a1 = a1.transformed(pt_e[0], indices=[0,1,2,3,4], projection=[0,1,2,3,4])
a1 = a1.transformed(pt_o[1], indices=[0,1,2,3,4], projection=[0,1,2,3,4])
rho1=g.copy(rho)
params2=[rho1]

a1.assert_gradient_error(rng, U + params2, U + params2, 1e-4, 1e-7)


# now test the local log-det-jacobian
act1 = g.qcd.gauge.smear.local_stout(rho=0.12, dimension=0, checkerboard=g.odd).action_log_det_jacobian()
act2 = pt_o[0].action_log_det_jacobian()

v1 = act1(U)
v2 = act2(U + params)
eps = abs(v1 - v2) / abs(v1)
g.message(f"Log-det-jacobian agreement: {eps}")
assert eps < 1e-13


# finally check the force terms of the log-det-jacobian
act2.assert_gradient_error(rng, U + params, U, 1e-4, 1e-7)

