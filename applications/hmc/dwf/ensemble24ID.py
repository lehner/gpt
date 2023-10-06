#!/usr/bin/env python3
import gpt as g
import numpy as np
import os, sys

rng = g.random("test")

# cold start
U = g.qcd.gauge.unit(g.grid([24, 24, 24, 64], g.double))

latest_it = None
it0 = 0
dst = g.default.get("--root", None)
N = 4000
for it in range(N):
    if os.path.exists(f"{dst}/ckpoint_lat.{it}"):
        latest_it = it

if latest_it is not None:
    g.copy(U, g.load(f"{dst}/ckpoint_lat.{latest_it}"))
    rng = g.random(f"test{dst}{latest_it}", "vectorized_ranlux24_24_64")
    it0 = latest_it + 1


pc = g.qcd.fermion.preconditioner
inv = g.algorithms.inverter
eofa_ratio = g.qcd.pseudofermion.action.exact_one_flavor_ratio


def two_flavor_ratio(fermion, m1, m2, solver):
    M1 = fermion(m1, m1)
    M2 = fermion(m2, m2)
    return g.qcd.pseudofermion.action.two_flavor_ratio_evenodd_schur([M1, M2], solver)


def quark(U0, m_plus, m_minus):
    return g.qcd.fermion.mobius(
        U0,
        mass_plus=m_plus,
        mass_minus=m_minus,
        M5=1.8,
        b=2.5,
        c=1.5,
        Ls=24,
        boundary_phases=[1, 1, 1, -1],
    )


def quark_tm(U0, m_plus, m_minus):
    assert m_plus == m_minus
    return g.qcd.fermion.wilson_twisted_mass(
        U0, mass=-1.8, mu=m_plus, boundary_phases=[1, 1, 1, -1]
    )


pc = g.qcd.fermion.preconditioner
inv = g.algorithms.inverter
sympl = g.algorithms.integrator.symplectic

F_grid_eo = quark(U, 1, 1).F_grid_eo
tmF_grid_eo = quark_tm(U, 1, 1).F_grid_eo


sloppy_prec = 1e-8
exact_prec = 1e-10

cg_s_inner = inv.cg({"eps": 1e-4, "eps_abs": sloppy_prec * 0.3, "maxiter": 40000, "miniter": 50})

cg_e_inner = inv.cg({"eps": 1e-4, "eps_abs": exact_prec * 0.3, "maxiter": 40000, "miniter": 50})

cg_s = inv.defect_correcting(
    inv.mixed_precision(cg_s_inner, g.single, g.double),
    eps=sloppy_prec,
    maxiter=100,
)

cg_e = inv.defect_correcting(
    inv.mixed_precision(cg_e_inner, g.single, g.double),
    eps=exact_prec,
    maxiter=100,
)

cg_e_tm = inv.defect_correcting(
    inv.mixed_precision(cg_e_inner, g.single, g.double),
    eps=exact_prec,
    maxiter=100,
)
cg_s_tm = cg_e_tm  # for simplicity just do exact solves


# chronological inverter
def mk_chron(slv):
    solution_space = []
    return inv.solution_history(
        solution_space,
        inv.sequence(inv.subspace_minimal_residual(solution_space), slv),
        10,
    )


def mk_slv_e():
    return mk_chron(
        inv.defect_correcting(
            inv.mixed_precision(inv.preconditioned(pc.eo2_ne(), cg_e_inner), g.single, g.double),
            eps=exact_prec,
            maxiter=100,
        )
    )


def mk_slv_s():
    return mk_chron(
        inv.defect_correcting(
            inv.mixed_precision(inv.preconditioned(pc.eo2_ne(), cg_s_inner), g.single, g.double),
            eps=sloppy_prec,
            maxiter=100,
        )
    )


# conjugate momenta
U_mom = g.group.cartesian(U)
rng.normal_element(U_mom)

action_gauge_mom = g.qcd.scalar.action.mass_term()
action_gauge = g.qcd.gauge.action.iwasaki(1.633)

rat = g.algorithms.rational.zolotarev_inverse_square_root(1.0**0.5, 12**0.5, 7)
rat_fnc = g.algorithms.rational.rational_function(rat.zeros, rat.poles, rat.norm)

# see params.py for parameter motivation
hasenbusch_ratios = [  # Nf=2+1
    (0.45, 1.0, None, two_flavor_ratio, mk_chron(cg_e), mk_chron(cg_s), quark),
    (0.18, 0.45, None, two_flavor_ratio, mk_chron(cg_e), mk_chron(cg_s), quark),
    (0.07, 0.18, None, two_flavor_ratio, mk_chron(cg_e), mk_chron(cg_s), quark),
    (0.017, 0.07, None, two_flavor_ratio, mk_chron(cg_e), mk_chron(cg_s), quark),
    (0.00107, 0.017, None, two_flavor_ratio, mk_chron(cg_e), mk_chron(cg_s), quark),
    (0.02, 0.07, None, two_flavor_ratio, cg_e_tm, cg_s_tm, quark_tm),
    (0.07, 0.18, None, two_flavor_ratio, cg_e_tm, cg_s_tm, quark_tm),
    (0.18, 0.5, None, two_flavor_ratio, cg_e_tm, cg_s_tm, quark_tm),
    (0.0850, 1.0, rat_fnc, eofa_ratio, mk_slv_e(), mk_slv_s(), quark),
]

fields = [
    (U + [g.vspincolor(F_grid_eo)]),
    (U + [g.vspincolor(F_grid_eo)]),
    (U + [g.vspincolor(F_grid_eo)]),
    (U + [g.vspincolor(F_grid_eo)]),
    (U + [g.vspincolor(F_grid_eo)]),
    (U + [g.vspincolor(tmF_grid_eo)]),
    (U + [g.vspincolor(tmF_grid_eo)]),
    (U + [g.vspincolor(tmF_grid_eo)]),
    (U + [g.vspincolor(U[0].grid)]),
]


# exact actions
action_fermions_e = [
    af(lambda m_plus, m_minus, q=qq: q(U, m_plus, m_minus), m1, m2, se)
    for m1, m2, rf, af, se, ss, qq in hasenbusch_ratios
]

# sloppy actions
action_fermions_s = [
    af(lambda m_plus, m_minus, q=qq: q(U, m_plus, m_minus), m1, m2, ss)
    for m1, m2, rf, af, se, ss, qq in hasenbusch_ratios
]


metro = g.algorithms.markov.metropolis(rng)

pure_gauge = True


if False:
    # power_iteration: iteration 39: 11.592318180179936
    g.default.push_verbose("power_iteration_convergence", True)
    sr = g.algorithms.eigen.power_iteration(eps=1e-5, real=True, maxiter=40)(
        action_fermions_s[-1].matrix(fields[-1]), rng.normal(fields[-1][-1])
    )
    g.default.pop_verbose()
    g.message("Spectral range", sr)


def hamiltonian(draw):
    if draw:
        rng.normal_element(U_mom)
        s = action_gauge(U)
        if not pure_gauge:
            # sp = sd(fields)
            for i in range(len(hasenbusch_ratios)):
                if hasenbusch_ratios[i][3] is eofa_ratio:
                    si = action_fermions_e[i].draw(fields[i], rng, hasenbusch_ratios[i][2])

                    # si = sp[i]
                    si_check = action_fermions_e[i](fields[i])
                    g.message("action", i, si_check)

                    r = f"{hasenbusch_ratios[i][0]}/{hasenbusch_ratios[i][1]}"
                    e = abs(si / si_check - 1)

                    g.message(f"Error of rational approximation for Hasenbusch ratio {r}: {e}")
                else:
                    si = action_fermions_e[i].draw(fields[i], rng)
                s += si
        h = s + action_gauge_mom(U_mom)
    else:
        s = action_gauge(U)
        if not pure_gauge:
            for i in range(len(hasenbusch_ratios)):
                s += action_fermions_e[i](fields[i])
        h = s + action_gauge_mom(U_mom)
    return h, s


log = sympl.log()


def fermion_force():
    x = [g.group.cartesian(u) for u in U]
    for y in x:
        y[:] = 0

    if not pure_gauge:
        forces = [[g.lattice(y) for y in x] for i in fields]

        log.time("fermion forces")
        for i in range(len(hasenbusch_ratios)):
            forces[i] = action_fermions_s[i].gradient(fields[i], fields[i][0 : len(U)])
        log.time()

        for i in range(len(hasenbusch_ratios)):
            log.gradient(forces[i], f"{hasenbusch_ratios[i][0]}/{hasenbusch_ratios[i][1]} {i}")
            for j in range(len(x)):
                x[j] += forces[i][j]
    return x


iq = sympl.update_q(U, log(lambda: action_gauge_mom.gradient(U_mom, U_mom), "gauge_mom"))

ip_gauge = sympl.update_p(U_mom, log(lambda: action_gauge.gradient(U, U), "gauge"))
ip_fermion = sympl.update_p(U_mom, fermion_force)
ip_gauge_fg = sympl.update_p_force_gradient(U, iq, U_mom, ip_gauge, ip_gauge)
ip_fermion_fg = sympl.update_p_force_gradient(U, iq, U_mom, ip_fermion, ip_fermion)
mdint = sympl.OMF2_force_gradient(
    12, ip_fermion, sympl.OMF2_force_gradient(4, ip_gauge, iq, ip_gauge_fg), ip_fermion_fg
)


def hmc(tau):
    accrej = metro(U)
    h0, s0 = hamiltonian(True)
    mdint(tau)
    h1, s1 = hamiltonian(False)
    return [accrej(h1, h0), s1 - s0, h1 - h0]


accept, total = 0, 0
for it in range(it0, N):
    pure_gauge = it < 10
    g.message(pure_gauge)
    a, dS, dH = hmc(1.0)
    accept += a
    total += 1
    plaq = g.qcd.gauge.plaquette(U)
    g.message(f"HMC {it} has P = {plaq}, dS = {dS}, dH = {dH}, acceptance = {accept/total}")
    for x in log.grad:
        g.message(f"{x} force norm2/sites =", np.mean(log.get(x)), "+-", np.std(log.get(x)))
    g.message(f"Timing:\n{log.time}")
    if it % 10 == 0:
        # reset statistics
        log.reset()
        g.message("Reset log")
    g.save(f"{dst}/ckpoint_lat.{it}", U, g.format.nersc())
    # g.save(f"{dst}/ckpoint_lat.{it}", U)
