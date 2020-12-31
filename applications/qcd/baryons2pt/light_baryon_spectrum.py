#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#          Lorenzo Barca 2020
#   DRAFT
#
import gpt as g
import numpy as np


# load configuration
U = g.load("/p/project/ea_lqcd_r/barca/ensembles/A653r000/A653r000n1750")

g.mem_report()

# smear the gauge links
alpha, n_iter, max_iteration, accuracy = 2.5, 5, 19, 1e-20
g.message("Applying APE-link smearing: \n")
g.message("alpha = {}; n_iter = {}; max_iteration = {}; accuracy = {}".format(alpha, n_iter, max_iteration, accuracy))
params_ape = {"alpha": alpha, "max_iteration": max_iteration, "accuracy": accuracy}
U_ape = U
for i in range(n_iter):
    g.message(f"\nAPE-iter:{i}")
    U_ape = g.qcd.gauge.smear.new_ape(U_ape, params_ape)
g.message("Done with APE-link smearing")

# use the gauge configuration grid
grid = U[0].grid
Vol = np.array(grid.fdimensions) # eg [24, 24, 24, 48]
Nt = Vol[-1]
L = Vol[0]

# quark
w1 = g.qcd.fermion.wilson_clover(U,{
    "kappa" : 0.137,
    "csw_r" : 0.,
    "csw_t" : 0.,
    "xi_0" : 1,
    "nu" : 1,
    "isAnisotropic" : False,
    "boundary_phases" : [ 1.0, 1.0, 1.0, -1.0 ]
})


# create point source
g.message("Creating the point source")
src = g.mspincolor(grid)
g.create.point(src, [0, 0, 0, 26])
t_src = np.array((0., 0., 0., 0.))

def sanity_check(propagator, measurement):
    g.message(f"sanity check {measurement}")
    correlator = g.slice(g.eval(g.trace(g.adj(propagator) * propagator)), 3)
    for t, c in enumerate(correlator):
        g.message(t, c)

# sanity check
sanity_check(src, "src")

# momentum
moms = np.array(([-1,0,0,0], [0,-1,0,0], [0,0,-1,0],
[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0]), dtype=float)
mom = 2.0 * np.pi * (moms - t_src) / L
mom_list = [ "mom_-100", "mom_0-10", "mom_00-1", "mom_000", "mom_100", "mom_010", "mom_001" ]
P_sanity = g.exp_ixp(mom[3])

# smear the point source
kappa, steps = 0.25, 5
g.message(f"Applying Wuppertal smearing to the source: kappa = {kappa}; steps = {steps}")
dimensions = [0, 1, 2]
smear = g.create.smear.wuppertal(U_ape, kappa=kappa, steps=steps, dimensions=dimensions)
src_smeared = g(smear * src)
g.message("Done with Wuppertal smearing to the source")

# sanity check
sanity_check(src_smeared, "src_smeared")

# build solver using g5m and cg
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-12, "maxiter": 1000})
slv_eo2_1 = w1.propagator(inv.preconditioned(pc.eo2_ne(), cg))

# propagator1
g.message("Starting the inversion1")
quark_prop1 = g.mspincolor(grid)
quark_prop1 @= slv_eo2_1 * src_smeared
g.message("Done with the inversion1")

# sanity check
sanity_check(quark_prop1, "quark_prop1")

# smear the propagator1
kappa, steps = 0.25, 5
g.message(f"Applying Wuppertal smearing to the propagator: kappa = {kappa}; steps = {steps}")
dimensions = [0, 1, 2]
smear = g.create.smear.wuppertal(U_ape, kappa=kappa, steps=steps, dimensions=dimensions)
quark_prop1_smeared = g(smear * quark_prop1)

# sanity check 1
sanity_check(quark_prop1_smeared, "quark_prop1_smeared")

quark_prop2_smeared = quark_prop1_smeared
quark_prop3_smeared = quark_prop1_smeared 

suN=3
g.message("Baryon spectrum code")
params = {"su(n)": suN, "kappa" : 0.137}
data_file = "light_baryon_spectrum.h5"

g.qcd.hadron.light_baryon_spectrum.light_baryon_spectrum(data_file, quark_prop1_smeared, quark_prop2_smeared, quark_prop3_smeared, mom, mom_list, params)


