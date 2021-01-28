#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#          Lorenzo Barca 2020

import gpt as g
import numpy as np
from gpt.qcd.hadron import spinmatrices as spm
from gpt.qcd.hadron.twopointcontraction.baryonoctet import \
    contract_proton, contract_xi_zero, contract_sigma_plus, contract_lambda_naive, contract_lambda
from gpt.qcd.hadron.twopointcontraction.baryondecuplet import \
    contract_sigma_plus_star, contract_delta_plus, contract_xi_zero_star, contract_omega

# show available memory
g.mem_report()

parameter_file = g.default.get("--params", "params.txt")
params = g.params(parameter_file, verbose=True)

# load configuration
rng = g.random("test")
U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], g.double), rng)
g.mem_report()


# use the gauge configuration grid
grid = U[0].grid
vol = np.array(grid.fdimensions)
L, Nt = vol[0], vol[-1]


# smear gauge links
params_link_smear = params["APE"]
g.message("Start APE-link smearing...")
U_ape = U
for i in range(params["APE"]["APE_iter"]):
    U_ape = g.qcd.gauge.smear.ape(U_ape, params_link_smear)
g.message("Done APE-link smearing.")


def sanity_check(dst, measurement):
    g.message(f"sanity check {measurement}")
    correlator = g.slice(g.eval(g.trace(g.adj(dst) * dst)), 3)
    for t, c in enumerate(correlator):
        g.message(t, c)


# create point source
src_pos = params["source_position"]
g.message(f"Creating point source in {src_pos}...")
src = g.mspincolor(grid)
g.create.point(src, src_pos)
g.message("Done creating point source.")
sanity_check(src, f"point_src")


# smear point source
params_quark_src_smear = params["wuppertal_smear_source"]
smear = g.create.smear.wuppertal(U_ape, params_quark_src_smear)
g.message("Start Wuppertal smearing to the source...")
src_smeared = g(smear * src)
g.message("Done Wuppertal smearing.")
sanity_check(src_smeared, f"shell_src")
del src


# build solver using g5m and cg
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-12, "maxiter": 1000})


# quark solver setup
flavors = ["light", "strange"]
solver_setup = dict()
slv_eo2 = dict()
for flavor in flavors:
    solver_setup[flavor] = g.qcd.fermion.wilson_clover(U, params[f"wilson_{flavor}"])
    slv_eo2[flavor] = solver_setup[flavor].propagator(inv.preconditioned(pc.eo2_ne(), cg))

# propagator
propagator = dict()
for flavor in flavors:
    propagator[f"point_{flavor}"] = g.mspincolor(grid)
    propagator[f"point_{flavor}"] @= slv_eo2[flavor] * src_smeared
    sanity_check(propagator[f"point_{flavor}"], f"point_{flavor}_propgator")

# smear propagators
params_quark_dst_smear = params["wuppertal_smear_propagator"]
smear = g.create.smear.wuppertal(U_ape, params_quark_dst_smear)
g.message("Start Wuppertal smearings...")
for flavor in flavors:
    g.message("Start Wuppertal smearing to {flavor} propagator")
    propagator[f"shell_{flavor}"] = g(smear * propagator[f"point_{flavor}"])
    sanity_check(propagator[f"shell_{flavor}"], f"shell_{flavor}_propgator")
    del propagator[f"point_{flavor}"]
g.message("Done Wuppertal smearings.")


def get_all_momenta_with_mom2max(mom2max):
    tmp = []
    for z in range(-mom2max, mom2max + 1):
        for y in range(-mom2max, mom2max + 1):
            for x in range(-mom2max, mom2max + 1):
                if x**2 + y**2 + z**2 <= mom2max:
                    tmp.append([x, y, z])
    return np.array(tmp)


def fft_baryon(baryon_list, contracted_baryon, integer_momenta, source_position):
    lattice_momenta = np.zeros((integer_momenta.shape[0], 4), dtype=np.float64)
    lattice_momenta[:, :3] = integer_momenta * 2 * np.pi / L
    for ii, momentum in enumerate(lattice_momenta):
        phase = g.exp_ixp(-1 * momentum)
        source_phase = np.exp(1.0j * np.dot(momentum[:3], source_position[:3]))
        baryon_list.append(np.array(g.slice(phase * contracted_baryon, 3)[:]) * source_phase)


contractions = {
    "Nucleon": {"requires": ["light", "light"], "contraction": contract_proton},
    "Sigma": {"requires": ["light", "strange"], "contraction": contract_sigma_plus},
    "Xi": {"requires": ["light", "strange"], "contraction": contract_xi_zero},
    "Lambda": {"requires": ["light", "light", "strange"], "contraction": contract_lambda},
    "LambdaNaive": {"requires": ["light", "light", "strange"], "contraction": contract_lambda_naive},
    "Delta": {"requires": ["light", "light"], "contraction": contract_delta_plus},
    "SigmaStar": {"requires": ["light", "strange"], "contraction": contract_sigma_plus_star},
    "XiStar": {"requires": ["light", "strange"], "contraction": contract_xi_zero_star},
    "Omega": {"requires": ["strange"], "contraction": contract_omega},
}

polarizations = {
    "unpolarized": spm.positive_parity_unpolarized(),
    "x-polarized": spm.positive_parity_xpolarized(),
    "y-polarized": spm.positive_parity_ypolarized(),
    "z-polarized": spm.positive_parity_zpolarized(),
    "mix-polarized": spm.positive_parity_unpolarized() + spm.positive_parity_zpolarized()
}

baryon_list = [
    *[["Nucleon", pol] for pol in ["unpolarized", "x-polarized", "y-polarized", "z-polarized"]],
    *[["Sigma", pol] for pol in ["unpolarized", "x-polarized", "y-polarized", "z-polarized"]],
    *[["Xi", pol] for pol in ["unpolarized", "x-polarized", "y-polarized", "z-polarized"]],
    *[["Lambda", pol] for pol in ["unpolarized", "x-polarized", "y-polarized", "z-polarized"]],
    *[["LambdaNaive", pol] for pol in ["unpolarized", "x-polarized", "y-polarized", "z-polarized"]],
    ["Delta", "mix-polarized"],
    ["SigmaStar", "mix-polarized"],
    ["XiStar", "mix-polarized"],
    ["Omega", "mix-polarized"],
]


# momentum
mom2_max = params["mom2_max"]
integer_momenta = get_all_momenta_with_mom2max(mom2_max)

output = dict()
output["source_position"] = src_pos
output["source_smearing"] = src_smearing
output["sink_smearing"] = sink_smearing
output["max_squared_momentum"] = mom2_max
output["momenta"] = integer_momenta


for baryon, pol_name in baryon_list:
    g.message(f"Baryon 2pt functions: Contracting {pol_name} {baryon}...")

    args = []
    for flavor in contractions[baryon]["requires"]:
        args.append(propagator[f"shell_{flavor}"])
    args.append(polarizations[pol_name])

    baryon_list = list()
    contracted_baryon = g.eval(contractions[baryon]["contraction"](*args))
    fft_baryon(baryon_list, contracted_baryon, integer_momenta, src_pos)

    correlators = np.array(baryon_list)

    for ii, mom in enumerate(integer_momenta):
        g.message(f"mom:{mom}")
        for t in range(Nt):
            g.message(t, correlators[ii][t])

    correlators = np.roll(correlators, -src_pos[3], axis=1)
    output[f"{baryon}/{pol_name}"] = correlators

