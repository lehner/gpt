#!/usr/bin/env python3
import gpt as g
import numpy as np
import os, sys, shutil, socket, time

g.default.set_verbose("defect_correcting_convergence")
g.default.set_verbose("cg_log_convergence")

category = g.default.get("--category", None)
select = g.default.get("--select", None)


ensembles_S = {
    # 32 cubed 3 flavor Fine ensembles #  
    "32F3fl-1" : { "L" : [32]*3 + [48], "beta" :  2.41, "ml" : 0.0088, "ms" : 0.0176, "mc" : None, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 1200 },
    # "32F3fl-2" : { "L" : [32]*3 + [48], "beta" :  2.47, "ml" : 0.0088, "ms" : 0.0176, "mc" : None, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8 },
    "32F3fh-1" : { "L" : [32]*3 + [48], "beta" :  2.41, "ml" : 0.0176, "ms" : 0.0176, "mc" : None, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 1200 },
    "32F3fh-2" : { "L" : [32]*3 + [48], "beta" :  2.47, "ml" : 0.0176, "ms" : 0.0176, "mc" : None, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 1200 },
    "32F3fx-1" : { "L" : [32]*3 + [48], "beta" :  2.41, "ml" : 0.0176, "ms" : 0.0352, "mc" : None, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 1200 },
    # added this one to also get ms dependence

    # 32 cubed 4 flavor Fine ensembles #
    "32F4fc1-1" : { "L" : [32]*3 + [48], "beta" : 2.39, "ml" :  0.0176, "ms" : 0.0176, "mc" : 0.187, "Ls": 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 1200 },
    # "32F4fc1-2" : { "L" : [32]*3 + [48], "beta" : 2.45, "ml" :  0.0176, "ms" : 0.0176, "mc" : 0.187, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8 },
    # "32F4fc2-1" : { "L" : [32]*3 + [48], "beta" : 2.39, "ml" :  0.0176, "ms" : 0.0176, "mc" : 0.234, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8 },
    # "32F4fc2-2" : { "L" : [32]*3 + [48], "beta" : 2.45, "ml" :  0.0176, "ms" : 0.0176, "mc" : 0.234, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8 },
    # "32F4fc3-1" : { "L" : [32]*3 + [48], "beta" : 2.39, "ml" :  0.0176, "ms" : 0.0176, "mc" : 0.281, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8 },
    # "32F4fc3-2" : { "L" : [32]*3 + [48], "beta" : 2.45, "ml" :  0.0176, "ms" : 0.0176, "mc" : 0.281, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8 },

    # 48^4 3 flavor Fine ensembles #
    # "48F3fl-1" : { "L" : [48]*4, "beta" :  2.41, "ml" : 0.0088, "ms" : 0.0176, "mc" : None, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8 },
    # "48F3fl-2" : { "L" : [48]*4, "beta" :  2.47, "ml" : 0.0088, "ms" : 0.0176, "mc" : None, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8 },
    "48F3fh-1" : { "L" : [48]*4, "beta" :  2.41, "ml" : 0.0176, "ms" : 0.0176, "mc" : None, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 1200 },
    "48F3fh-2" : { "L" : [48]*4, "beta" :  2.47, "ml" : 0.0176, "ms" : 0.0176, "mc" : None, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 1200 },

    # 48^4 4 flavor Fine ensembles #
    # "48F4fl-1" : { "L" : [48]*4, "beta" : 2.39, "ml" :  0.0088, "ms" : 0.0176, "mc" : 0.187, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8 },
    # "48F4fl-2" : { "L" : [48]*4, "beta" : 2.45, "ml" :  0.0088, "ms" : 0.0176, "mc" : 0.187, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8 },
    "48F4fh-1" : { "L" : [48]*4, "beta" : 2.39, "ml" :  0.0176, "ms" : 0.0176, "mc" : 0.187, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 1200 },
    # "48F4fh-2" : { "L" : [48]*4, "beta" : 2.45, "ml" :  0.0176, "ms" : 0.0176, "mc" : 0.187, "Ls" : 12, "b" : 1.25, "c" : 0.25, "M5" : 1.8 },

    # 48^4 VF 3 flavor ensembles #
    # "48VF3fl-1" : { "L" : [48]*4, "beta" :  2.56, "ml" : 0.00625, "ms" : 0.0125, "mc" : None, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8 },
    # "48VF3fl-2" : { "L" : [48]*4, "beta" :  2.62, "ml" : 0.00625, "ms" : 0.0125, "mc" : None, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8 },
    "48VF3fh-1" : { "L" : [48]*4, "beta" :  2.56, "ml" : 0.0125, "ms" : 0.0125, "mc" : None, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 2400 },
    "48VF3fh-2" : { "L" : [48]*4, "beta" :  2.62, "ml" : 0.0125, "ms" : 0.0125, "mc" : None, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 2400 },

    # 48^4 VF 4 flavor ensembles #
    "48VF4fc1-1" : { "L" : [48]*4, "beta" :  2.54, "ml" : 0.0125, "ms" : 0.0125, "mc" : 0.142, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 2400 },
    # "48VF4fc1-2" : { "L" : [48]*4, "beta" :  2.60, "ml" : 0.0125, "ms" : 0.0125, "mc" : 0.142, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8 },
    "48VF4fc2-1" : { "L" : [48]*4, "beta" :  2.54, "ml" : 0.0125, "ms" : 0.0125, "mc" : 0.178, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 2400 },
    # "48VF4fc2-2" : { "L" : [48]*4, "beta" :  2.60, "ml" : 0.0125, "ms" : 0.0125, "mc" : 0.178, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8 },
    "48VF4fc3-1" : { "L" : [48]*4, "beta" :  2.54, "ml" : 0.0125, "ms" : 0.0125, "mc" : 0.213, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 2400 },
    # "48VF4fc3-2" : { "L" : [48]*4, "beta" :  2.60, "ml" : 0.0125, "ms" : 0.0125, "mc" : 0.213, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8 }, 
}

#ensembles_M = {
#}

ensembles_L = {
    # 64^3x96 3 flavor VF ensembles #
    # "64VF3fl-1" : { "L" : [64]*3 + [96], "beta" :  2.56, "ml" : 0.00625, "ms" : 0.0125, "mc" : None, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8 },
    # "64VF3fl-2" : { "L" : [64]*3 + [96], "beta" :  2.62, "ml" : 0.00625, "ms" : 0.0125, "mc" : None, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8 },
    "64VF3fh-1" : { "L" : [64]*3 + [96], "beta" :  2.56, "ml" : 0.0125, "ms" : 0.0125, "mc" : None, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 2400 },
    "64VF3fh-2" : { "L" : [64]*3 + [96], "beta" :  2.62, "ml" : 0.0125, "ms" : 0.0125, "mc" : None, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 2400 },

    # 64^3x96 4 flavor VF ensembles #
    # "64VF4fl-1" : { "L" : [64]*3 + [96], "beta" :  2.54, "ml" : 0.00625, "ms" : 0.0125, "mc" : 0.142, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8 },
    # "64VF4fl-2" : { "L" : [64]*3 + [96], "beta" :  2.60, "ml" : 0.00625, "ms" : 0.0125, "mc" : 0.142, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8 },
    "64VF4fh-1" : { "L" : [64]*3 + [96], "beta" :  2.54, "ml" : 0.0125, "ms" : 0.0125, "mc" : 0.142, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 2400 },
    # "64VF4fh-2" : { "L" : [64]*3 + [96], "beta" :  2.60, "ml" : 0.0125, "ms" : 0.0125, "mc" : 0.142, "Ls" : 12, "b" : 1.175, "c" : 0.175, "M5" : 1.8 },

    # 96^4 3 flavor SF ensembles #
    "96SF3f-1" : { "L" : [96]*4, "beta" :  2.71, "ml" : 0.0093 , "ms" : 0.0093, "mc" : None, "Ls" : 12, "b" : 1.125, "c" : 0.125, "M5" : 1.8, "nsteps" : 8, "nsubsteps" : 4, "tau" : 8, "nwf_max" : 4800 },
    # "96SF3f-2" : { "L" : [96]*4, "beta" :  2.77, "ml" : 0.0093, "ms" : 0.0093, "mc" : None, "Ls" : 12, "b" : 1.125, "c" : 0.125, "M5" : 1.8 },
}

#ensembles_XL = {
#}

ensembles = {
    "S" : ensembles_S,
    "L" : ensembles_L,
}[category]

if select is not None:
    ensembles = {
        select : ensembles[select]
    }

####
run_replicas = [0,1] # run with reproduction replica
conf_range = range(400)
pure_gauge = False # allows for first few trajectories to be run without fermions; don't use for now

def light(U0, m_plus, m_minus, params):
    return g.qcd.fermion.mobius(
        U0,
        mass_plus=m_plus,
        mass_minus=m_minus,
        M5=params["M5"],
        b=params["b"],
        c=params["c"],
        Ls=params["Ls"],
        boundary_phases=[1, 1, 1, -1],
    )

def two_flavor_ratio(fermion, m1, m2, solver):
    M1 = fermion(m1, m1)
    M2 = fermion(m2, m2)
    return g.qcd.pseudofermion.action.two_flavor_ratio_evenodd_schur([M1, M2], solver)



action_gauge_mom = g.qcd.scalar.action.mass_term()

rat = g.algorithms.rational.zolotarev_inverse_square_root(1.0**0.5, 70.0**0.5, 11)
rat_fnc = g.algorithms.rational.rational_function(rat.zeros, rat.poles, rat.norm)

#root_output = "/lus/flare/projects/LatticeFlavor/lehner/ensemble-NPR"
root_output = "ensemble-NPR"

pc = g.qcd.fermion.preconditioner
inv = g.algorithms.inverter
eofa_ratio = g.qcd.pseudofermion.action.exact_one_flavor_ratio

# FT
rho = 0.12
sm = [
    g.qcd.gauge.smear.local_stout(rho=rho, dimension=mu, checkerboard=p) for mu in range(4) for p in [g.even, g.odd]
]


pc = g.qcd.fermion.preconditioner
inv = g.algorithms.inverter
sympl = g.algorithms.integrator.symplectic

sloppy_prec = 1e-9
sloppy_prec_light = 1e-9
exact_prec = 1e-11

sloppy_prec = 1e-10
sloppy_prec_light = 1e-10
exact_prec = 1e-12

cg_s_inner = inv.cg({"eps": 1e-4, "eps_abs": sloppy_prec * 0.15, "maxiter": 40000, "miniter": 50, "fail_if_not_converged": True})
cg_s_light_inner = inv.cg({"eps": 1e-4, "eps_abs": sloppy_prec_light * 0.15, "maxiter": 40000, "miniter": 50, "fail_if_not_converged": True})
cg_e_inner = inv.cg({"eps": 1e-4, "eps_abs": exact_prec * 0.3, "maxiter": 40000, "miniter": 50, "fail_if_not_converged": True})

cg_s = inv.defect_correcting(
    inv.mixed_precision(cg_s_inner, g.single, g.double),
    eps=sloppy_prec,
    maxiter=100,
)

cg_s_light = inv.defect_correcting(
    inv.mixed_precision(cg_s_light_inner, g.single, g.double),
    eps=sloppy_prec_light,
    maxiter=100,
)

cg_e = inv.defect_correcting(
    inv.mixed_precision(cg_e_inner, g.single, g.double),
    eps=exact_prec,
    maxiter=100,
)

# chronological inverter
def mk_chron(slv, solution_space = None, N = 10, grid = None):
    if solution_space is None:
        solution_space = []
    return inv.solution_history(
        solution_space,
        inv.sequence(inv.subspace_minimal_residual(solution_space), slv),
        N,
    )


def mk_slv_e(solution_space, N, grid = None):
    return mk_chron(
        inv.defect_correcting(
            inv.mixed_precision(inv.preconditioned(pc.eo2_ne(), cg_e_inner), g.single, g.double),
            eps=exact_prec,
            maxiter=100,
        ), solution_space = solution_space, N=N
    )


def mk_slv_s(solution_space, N, grid = None):
    return mk_chron(
        inv.defect_correcting(
            inv.mixed_precision(inv.preconditioned(pc.eo2_ne(), cg_s_inner), g.single, g.double),
            eps=sloppy_prec,
            maxiter=100,
        ), solution_space = solution_space, N=N
    )


def transform(aa, s, i):
    aa_transformed = aa[i].transformed(s, indices=list(range(len(U))))
    aa_orig = aa[i]
    def draw(fields, rng, *extra):
        sfields = s(fields[0:len(U)]) + fields[len(U):]
        x = aa_orig.draw(sfields, rng, *extra)
        return x
    aa_transformed.draw = draw
    aa[i] = aa_transformed



def setup(tag):
    global U, U_mom, action_gauge, hasenbusch_ratios, fields, action_fermions_e, action_fermions_s, ensemble_tag, css, a_log_det, mdint

    params = ensembles[tag]
    
    U = g.qcd.gauge.unit(g.grid(params["L"], g.double))
    U_mom = g.group.cartesian(U)
    action_gauge = g.qcd.gauge.action.iwasaki(params["beta"])

    lq = light(U, 1, 1, params)
    F_grid_eo = lq.F_grid_eo
    F_grid = lq.F_grid

    css = [
        ([], 8, F_grid_eo),
        ([], 16, F_grid_eo),
        ([], 4, F_grid)
    ]

    m_l = params["ml"]
    m_s = params["ms"]
    m_c = params["mc"]
    b = params["b"]
    c = params["c"]

    fields = [
        (U + [g.vspincolor(F_grid_eo)]),
        (U + [g.vspincolor(F_grid_eo)]),
        (U + [g.vspincolor(F_grid_eo)]),
        (U + [g.vspincolor(F_grid_eo)]),
        (U + [g.vspincolor(U[0].grid)]),
        (U + [g.vspincolor(U[0].grid)]),
    ]
    
    hasenbusch_ratios = [
        (0.65, 1.0, None, two_flavor_ratio, mk_chron(cg_e, *css[0]), mk_chron(cg_s, *css[0]), light),
        (0.28, 0.65, None, two_flavor_ratio, mk_chron(cg_e, *css[0]), mk_chron(cg_s, *css[0]), light),
        (0.11, 0.28, None, two_flavor_ratio, mk_chron(cg_e, *css[0]), mk_chron(cg_s, *css[0]), light),
        (m_l, 0.11, None, two_flavor_ratio, mk_chron(cg_e, *css[1]), mk_chron(cg_s_light, *css[1]), light),
        (0.23, 1.0, rat_fnc, eofa_ratio, mk_slv_e(*css[2]), mk_slv_s(*css[2]), light),
        (m_s, 0.23, rat_fnc, eofa_ratio, mk_slv_e(*css[2]), mk_slv_s(*css[2]), light),
    ]

    if m_c is not None:
        css.append(
            ([], 4, F_grid)
        )
        
        hasenbusch_ratios.append(
            (m_c, 1.0, rat_fnc, eofa_ratio, mk_slv_e(*css[3]), mk_slv_s(*css[3]), light)
        )

        fields.append(
            (U + [g.vspincolor(U[0].grid)])
        )

    ensemble_tag = tag
    
    # exact actions
    action_fermions_e = [
        af(lambda m_plus, m_minus, qqi=qq: qqi(U, m_plus, m_minus, params), m1, m2, se)
        for m1, m2, rf, af, se, ss, qq in hasenbusch_ratios
    ]
    
    # sloppy actions
    action_fermions_s = [
        af(lambda m_plus, m_minus, qqi=qq: qqi(U, m_plus, m_minus, params), m1, m2, ss)
        for m1, m2, rf, af, se, ss, qq in hasenbusch_ratios
    ]

        
    a_log_det = None
    for s in sm: # sm = [ Sx Sy Sz St ] ->   f(U) -> f(Sx(U)) -> ... -> f(Sx(Sy(Sz(St(U)))))
        action_gauge = action_gauge.transformed(s)
        for i in range(len(hasenbusch_ratios)):
            transform(action_fermions_e, s, i)
            transform(action_fermions_s, s, i)
        if a_log_det is None:
            a_log_det = s.action_log_det_jacobian()
        else:
            a_log_det = a_log_det.transformed(s) + s.action_log_det_jacobian()

    if False:
        g.default.push_verbose("power_iteration_convergence", True)
        sr = g.algorithms.eigen.power_iteration(eps=1e-5, real=True, maxiter=40)(
            action_fermions_s[-1].matrix(fields[-1]), g.random("test").normal(fields[-1][-1])
        )
        g.default.pop_verbose()
        g.message("Spectral range", sr)
        sys.exit(0)


    iq = sympl.update_q(
        U, log(lambda: action_gauge_mom.gradient(U_mom, U_mom), "gauge_mom")
    )
    
    ip_gauge = sympl.update_p(U_mom, gauge_force)
    ip_fermion = sympl.update_p(U_mom, fermion_force, tag="Q_fermion")
    ip_log_det = sympl.update_p(U_mom, log_det_force)
    ip_log_det_sp = sympl.update_p(U_mom, log_det_force_sp)
    
    ip_gauge_fg = sympl.update_p_force_gradient(U, iq, U_mom, ip_gauge, ip_gauge)
    ip_fermion_fg = sympl.update_p_force_gradient(U, iq, U_mom, ip_fermion, ip_fermion, tag="Q_fg_fermion")
    ip_log_det_fg = sympl.update_p_force_gradient(U, iq, U_mom, ip_log_det, ip_log_det)#_sp
    
    mdint = sympl.OMF2_force_gradient(
        params["nsubsteps"], ip_fermion,
        sympl.OMF2_force_gradient(2, ip_log_det,
                                  sympl.OMF2_force_gradient(2, ip_gauge, iq, ip_gauge_fg),
                                  ip_log_det_fg),
        ip_fermion_fg
    )
    
    g.message(mdint)

    g.message("Adopted setup for", ensemble_tag)



# initial config
for tag in ensembles:
    fnd = f"{root_output}/{tag}" 
    fn = f"{fnd}/ckpoint_lat.0"
    if not os.path.exists(fn):
        if g.rank() == 0 and not os.path.exists(fnd):
            os.makedirs(fnd, exist_ok=True)
        g.barrier()
        g.save(fn, g.qcd.gauge.random(g.grid(ensembles[tag]["L"], g.double), g.random("init" + tag), scale=0.1), g.format.nersc())



def hamiltonian(draw, rng):
    ald = a_log_det(U)
    if draw:
        rng.normal_element(U_mom)

        s = action_gauge(U)
        g.message("Gluonic action", s)
        g.message("Log-det action", ald)
        if not pure_gauge:
            # sp = sd(fields)
            for i in range(len(hasenbusch_ratios)):
                if hasenbusch_ratios[i][3] is not two_flavor_ratio:
                    si = action_fermions_e[i].draw(fields[i], rng, hasenbusch_ratios[i][2])

                    # si = sp[i]
                    si_check = action_fermions_e[i](fields[i])
                    g.message("action", i, si_check)

                    r = f"{hasenbusch_ratios[i][0]}/{hasenbusch_ratios[i][1]}"
                    e = abs(si / si_check - 1)

                    g.message(f"Error of rational approximation for Hasenbusch ratio {r}: {e} or absolute {si-si_check}")
                else:
                    si = action_fermions_e[i].draw(fields[i], rng)
                s += si
        h = s + action_gauge_mom(U_mom) + ald
    else:
        s = action_gauge(U)
        if not pure_gauge:
            for i in range(len(hasenbusch_ratios)):
                sa = action_fermions_e[i](fields[i])
                g.message(f"Calculate Hamiltonian for {hasenbusch_ratios[i][0]}/{hasenbusch_ratios[i][1]} = {sa}")
                s += sa
        h = s + action_gauge_mom(U_mom) + ald
    return h, s




log = sympl.log()

ff_iterator = 0
def fermion_force():
    global ff_iterator
    g.message(f"Fermion force {ff_iterator}")
    ff_iterator += 1
    x = [g.group.cartesian(u) for u in U]
    for y in x:
        y[:] = 0

    g.mem_report(details=False)
    if not pure_gauge:
        forces = [[g.lattice(y) for y in x] for i in fields]

        log.time("fermion forces")
        for i in range(len(hasenbusch_ratios)):
            g.message(f"Hasenbusch ratio {hasenbusch_ratios[i][0]}/{hasenbusch_ratios[i][1]}")
            forces[i] = action_fermions_s[i].gradient(fields[i], fields[i][0 : len(U)])
            g.message("Ratio complete")
            g.mem_report(details=False)

        g.message("Log Time")
        log.time()

        g.message("Add and log forces")
        for i in range(len(hasenbusch_ratios)):
            log.gradient(forces[i], f"{hasenbusch_ratios[i][0]}/{hasenbusch_ratios[i][1]} {i}")
            for j in range(len(x)):
                x[j] += forces[i][j]
        g.message("Done")

    g.message(f"Fermion force done")
    return x

def gauge_force():
    g.message("Compute gauge force")
    x = log(lambda: action_gauge.gradient(U, U), "gauge")()
    g.message("first level force complete")
    return x

def log_det_force():
    g.message("Compute log_det force")
    x = log(lambda: a_log_det.gradient(U, U), "log det")()
    g.message("second level force complete")
    return x

def log_det_force_sp():
    g.message("Compute log_det force (sp)")
    Usp = g.convert(U, g.single)
    x = log(lambda: g.convert(a_log_det.gradient(Usp, Usp), g.double), "log det")()
    g.message("second level force complete (sp)")
    return x

if False:
    g.default.set_verbose("stencil_performance")
    g.default.set_verbose("stout_performance")
    g.default.set_verbose("auto_tune")
    log_det_force()
    sys.exit(0)


################################################################################
# Job - general reproduction
################################################################################
class job_reproduction_base(g.jobs.base):
    def __init__(self, tag, output_files, replica, dependencies):
        self.reproduction_tag = tag
        self.output_files = output_files
        self.replica = replica
        super().__init__(f"{tag}/{replica}", dependencies)

    def purge(self, root):
        if g.rank() == 0:
            if os.path.exists(f"{root}/{self.name}/hosts"):
                os.unlink(f"{root}/{self.name}/hosts")
        super().purge(root)

    def perform(self, root):
        # now save list of nodes on which this job was performed
        # need to make sure that when releasing a reproduction job
        # that it was done with non-overlapping nodes
        hosts = []
        hostname = socket.gethostname()
        for r in range(g.ranks()):
            hosts.append(g.broadcast(r, hostname))
        if g.rank() == 0:
            np.savetxt(f"{root}/{self.name}/hosts", hosts, fmt='%s')

        # fingerprinting support
        for i in range(1000):
            fn = f"{root}/fingerprints/{self.name}/{i}"
            if not os.path.exists(fn):
                break

        g.message(f"Fingerprints saved in {fn}")
        g.fingerprint.start(fn)
        
        # and perform task
        self.perform_inner(root)

        g.fingerprint.flush()


################################################################################
# Job - checkpoint
# Establishes that checkpointed configuration is healthy
################################################################################
class job_checkpoint(job_reproduction_base):
    def __init__(self, stream, conf, replica, dependencies):
        self.stream = stream
        self.conf = conf
        super().__init__(
            f"{stream}/{conf}_checkpoint",
            ["config.0"],
            replica,
            dependencies
        )
        self.weight = 0.1

    def perform_inner(self, root):
        fn = f"{root}/{self.stream}/ckpoint_lat.{self.conf}"
        if os.path.exists(fn):
            try:
                next_U = g.load(fn)
            except:
                return False

            # check unitarity
            for mu in range(4):
                eps2 = g.norm2(next_U[mu] * g.adj(next_U[mu]) - g.identity(next_U[mu])) / g.norm2(next_U[mu])
                g.message("Unitarity defect for mu=",mu,"is",eps2)
                assert eps2 < 1e-25

                eps2 = g.norm2(g.matrix.det(next_U[mu]) - g.identity(g.complex(next_U[0].grid))) / next_U[0].grid.gsites
                g.message("Determinant defect for mu=",mu,"is",eps2)
                assert eps2 < 1e-25
                
                g.message("Project")
                g.project(next_U[mu], "defect_left")

                eps2 = g.norm2(next_U[mu] * g.adj(next_U[mu]) - g.identity(next_U[mu])) / g.norm2(next_U[mu])
                g.message("Unitarity defect for mu=",mu,"is",eps2)
                assert eps2 < 1e-28

                eps2 = g.norm2(g.matrix.det(next_U[mu]) - g.identity(g.complex(next_U[0].grid))) / next_U[0].grid.gsites
                g.message("Determinant defect for mu=",mu,"is",eps2)
                assert eps2 < 1e-28
                
            fn = f"{root}/{self.name}/config.0"
            g.save(fn, next_U)
            try:
                next_U = g.load(fn)
            except:
                return False
            if g.rank() == 0:
                f = open(f"{root}/{self.name}/loaded", "wt")
                f.close()
        g.barrier()

    def check(self, root):
        return os.path.exists(f"{root}/{self.name}/loaded")


################################################################################
# Job - reproduction verify
################################################################################
class job_reproduction_verify(g.jobs.base):
    def __init__(self, replica_jobs):
        rep = replica_jobs[0]
        super().__init__(f"{rep.reproduction_tag}/verify", [j.name for j in replica_jobs])
        self.weight = 1.0
        self.replica_jobs = replica_jobs
        self.flog = None
            
    def log(self, root, msg):
        if g.rank() == 0:
            if self.flog is None:
                self.flog = open(f"{root}/{self.name}/log", "at")
            self.flog.write(f"{msg}\n")
            self.flog.flush()

    def recursive_verify(self, root, A, B):
        if isinstance(A, list):
            return all([ self.recursive_verify(root, a, b) for a, b in zip(A, B) ])
        elif isinstance(A, (float, complex)):
            self.log(root, f"Verify number {A} == {B}")
            return A == B
        elif isinstance(A, g.lattice):
            eps2 = g.norm2(A - B)
            g.message("Result", eps2, g.norm2(A))
            self.log(root, f"Verify lattice {A.otype} {A.grid}: {eps2}")
            return eps2 == 0.0
        else:
            raise Exception("Invalid type",type(A))

        
    def perform(self, root):
        # test the replicas
        verified = True
        ref = self.replica_jobs[0]
        for rep in self.replica_jobs[1:]:
            for of in rep.output_files:
                fn_ref = f"{root}/{self.name}/../{ref.replica}/{of}"
                fn_rep = f"{root}/{self.name}/../{rep.replica}/{of}"
                g.message("Check", fn_ref, fn_rep)
                A = g.util.to_list(g.load(fn_ref))
                B = g.util.to_list(g.load(fn_rep))
                if not self.recursive_verify(root, A, B):
                    self.log(root, f"Failure to verify file {fn_ref} against {fn_rep}")
                    self.log(root, open(f"{root}/{self.name}/../{ref.replica}/.started").read())
                    self.log(root, open(f"{root}/{self.name}/../{rep.replica}/.started").read())
                    verified = False

        if verified:
            if g.rank() == 0:
                open(f"{root}/{self.name}/verified", "wt").close()
        else:
            # verify failed, purge the jobs
            for rep in self.replica_jobs:
                rep.purge(root)

            if g.rank() == 0:
                open(f"{root}/reproduction.failures", "at").write(
                    open(f"{root}/{self.name}/log").read()
                )

        g.barrier()

        
    def check(self, root):
        return os.path.exists(f"{root}/{self.name}/verified")

################################################################################
# Job - release disk space
################################################################################
class job_release(g.jobs.base):
    def __init__(self, directories, job_ver):
        self.directories = directories
        super().__init__(f"{job_ver.name}/../release", [job_ver.name])
        self.weight = 1.0

    def log(self, root, msg):
        if g.rank() == 0:
            f = open(f"{root}/{self.name}/log", "at")
            f.write(f"{msg}\n")
            f.close()
        
    def perform(self, root):
        for dn in self.directories:
            self.log(root, f"Remove {dn}")
            if g.rank() == 0:
                if os.path.exists(dn):
                    shutil.rmtree(dn)
        g.barrier()
        
    def check(self, root):
        return True
    

################################################################################
# Job - draw
################################################################################
class job_draw(job_reproduction_base):
    def __init__(self, stream, conf, replica, dependencies):
        self.stream = stream
        self.conf = conf
        super().__init__(
            f"{stream}/{conf}_draw",
            ["state.0","state.draw"],
            replica,
            dependencies
        )
        self.weight = 1.0

    def perform_inner(self, root):
        global U

        setup(self.stream)
        
        next_U = g.load(f"{root}/{self.stream}/{self.conf}_checkpoint/0/config.0")
        # undo the smearing
        U0 = g.copy(next_U)
        for s in sm:
            next_U = s.inv(next_U)
        g.copy(U, next_U)
        U1 = g.copy(next_U)
        for s in reversed(sm):
            U1 = s(U1)

        for mu in range(4):
            eps2 = g.norm2(U1[mu] - U0[mu]) / g.norm2(U0[mu])
            g.message(f"Test s(s.inv) = id, U_{mu} = {eps2}")
            assert eps2 < 1e-25

        g.message("Plaquette of integration field",g.qcd.gauge.plaquette(U))

        # reset css
        for c in css:
            c[0].clear()

        # rng
        rng = g.random(f"{self.stream}/{self.conf}", "vectorized_ranlux24_24_64")
        h0, s0 = hamiltonian(True, rng)

        g.save(
            f"{root}/{self.name}/state.0",
            [
                [c[0] for c in css],
                U,
                U_mom,
            ]
        )

        g.save(
            f"{root}/{self.name}/state.draw",
            [
                h0,
                s0,
                [x[-1] for x in fields]
            ]
        )
    
    def check(self, root):
        return True


################################################################################
# Job - molecular dynamics
################################################################################
class job_md(job_reproduction_base):
    def __init__(self, stream, conf, step, replica, dependencies):
        self.stream = stream
        self.conf = conf
        self.step = step
        super().__init__(
            f"{stream}/{conf}_md_{step}",
            ["state.0"],
            replica,
            dependencies
        )
        self.weight = 1.0

    def perform_inner(self, root):
        global U, U_mom, css

        setup(self.stream)
        
        h0, s0, flds = g.load(f"{root}/{self.stream}/{self.conf}_draw/0/state.draw")

        for i in range(len(fields)):
            fields[i][-1] @= flds[i]

        del flds

        if self.step == 0:
            fn = f"{root}/{self.stream}/{self.conf}_draw/0/state.0"
        else:
            fn = f"{root}/{self.stream}/{self.conf}_md_{self.step-1}/0/state.0"

        l_css, l_U, l_U_mom = g.load(fn)

        for i in range(len(css)):
            css[i][0].clear()
            css[i][0].extend(l_css[i])
            l_css[i] = None
            
        for i in range(4):
            U[i] @= l_U[i]
            U_mom[i] @= l_U_mom[i]

        del l_css
        del l_U
        del l_U_mom

        # mdint
        mdint(ensembles[self.stream]["tau"] / ensembles[self.stream]["nsteps"])

        g.save(
            f"{root}/{self.name}/state.0",
            [
                [c[0] for c in css],
                U,
                U_mom,
            ]
        )
        
    def check(self, root):
        return True


################################################################################
# Job - calculate Hamiltonian
################################################################################
class job_hamiltonian(job_reproduction_base):
    def __init__(self, stream, conf, step, replica, dependencies):
        self.stream = stream
        self.conf = conf
        self.step = step
        super().__init__(
            f"{stream}/{conf}_H_{step}",
            ["H"],
            replica,
            dependencies
        )
        self.weight = 1.0

    def perform_inner(self, root):
        global U, U_mom, css

        setup(self.stream)
        
        h0, s0, flds = g.load(f"{root}/{self.stream}/{self.conf}_draw/0/state.draw")

        for i in range(len(fields)):
            fields[i][-1] @= flds[i]

        del flds

        fn = f"{root}/{self.stream}/{self.conf}_md_{self.step}/0/state.0"

        l_css, l_U, l_U_mom = g.load(fn)

        for i in range(len(css)):
            css[i][0].clear()
            css[i][0].extend(l_css[i])
            l_css[i] = None
            
        for i in range(4):
            U[i] @= l_U[i]
            U_mom[i] @= l_U_mom[i]

        del l_css
        del l_U
        del l_U_mom

        # mdint
        h1, s1 = hamiltonian(False, g.random("none"))

        g.save(
            f"{root}/{self.name}/H",
            [h0,h1,h1-h0,s0,s1]
        )
        
    def check(self, root):
        return True


################################################################################
# Job - write checkpoint
################################################################################
class job_write_checkpoint(job_reproduction_base):
    def __init__(self, stream, conf, step, name, replica, dependencies):
        self.stream = stream
        self.conf = conf
        self.step = step
        self.tag = name
        super().__init__(
            f"{stream}/{conf}_write_checkpoint_{step}",
            ["config"],
            replica,
            dependencies
        )
        self.weight = 1.0

    def perform_inner(self, root):
        global U, U_mom, css

        fn = f"{root}/{self.stream}/{self.conf}_md_{self.step}/0/state.0"

        l_css, l_U, l_U_mom = g.load(fn)

        Uft = l_U
        for s in reversed(sm):
            Uft = s(Uft)

        fn = f"{root}/{self.stream}/{self.conf}_H_{self.step}/0/H"
        H = g.load(fn)
        dH = H[2]
        
        plaq = g.qcd.gauge.plaquette(l_U)
        plaqft = g.qcd.gauge.plaquette(Uft)
        if g.rank() == 0:
            flog = open(f"{root}/{self.stream}/ckpoint_lat.{self.tag}.log","wt")
            flog.write(f"dH {dH}\n")
            flog.write(f"P {plaqft}\n")
            flog.write(f"Pft {plaq}\n")
            flog.close()

        if self.replica == 0:
            g.save(
                f"{root}/{self.stream}/config.{self.tag}",
                Uft
            )

        g.save(f"{root}/{self.name}/config", Uft)

        if self.replica == 0:
            g.save(
                f"{root}/{self.stream}/ckpoint_lat.{self.tag}",
                Uft,
                g.format.nersc()
            )

        A = g.load(f"{root}/{self.stream}/ckpoint_lat.{self.tag}")
        B = g.load(f"{root}/{self.name}/config")
        for mu in range(4):
            eps2 = g.norm2(A[mu] - B[mu])
            g.message("CHECK", eps2)
            assert eps2 == 0.0

    def check(self, root):
        return True



################################################################################
# Job - gluonic measurements
################################################################################
class job_measure_glue(job_reproduction_base):
    def __init__(self, stream, conf, name, replica, dependencies):
        self.stream = stream
        self.conf = conf
        self.tag = name
        super().__init__(
            f"{stream}/{conf}_measure_glue",
            ["glue"],
            replica,
            dependencies
        )
        self.weight = 1.0

    def perform_inner(self, root):
        global U, U_mom, css

        config = f"{root}/{self.stream}/ckpoint_lat.{self.tag}"

        eps = 0.01
        nsteps = ensembles[self.stream]["nwf_max"]
        ntop = 200

        U = g.load(config)

        vol3d = float(np.prod(U[0].grid.gdimensions[0:3]))

        if self.replica != 0:
            config = config + "." + str(self.replica)

        res = []
        
        w = g.corr_io.writer(f"{config}.gluonic")

        # first plaquette
        g.message("Plaquette")
        P = g.slice(g.qcd.gauge.rectangle(U, 1, 1, field=True) / vol3d, 3)
        w.write("P", P)
        res.append(P)
        
        # test wilson flow
        U_wf = U
        U_wf = g.qcd.gauge.smear.wilson_flow(U_wf, epsilon=eps)
        U_wf = g.qcd.gauge.smear.wilson_flow(U_wf, epsilon=-eps)
        g.message(f"Test result: {(g.norm2(U[0]-U_wf[0])/g.norm2(U[0]))**0.5}")

        if g.rank() == 0:
            fQ = open(f"{config}.Q","wt")
            fE = open(f"{config}.E","wt")
            ft0 = open(f"{config}.t0","wt")
        else:
            fQ = None
            fE = None
            ft0 = None
        
        tau = 0.0
        U_wf = U
        c = {}
        for i in range(nsteps):
        
            U_wf = g.qcd.gauge.smear.wilson_flow(U_wf, epsilon=eps)
            tau += eps
            g.message("%g" % tau)

            g.message("Field Strength")
            E = g.slice(g.qcd.gauge.energy_density(U_wf, field=True) / vol3d, 3)
            w.write("E(%g)" % tau, E)
            res.append(E)

            E = sum(E).real / len(E)
            t2E = tau**2 * E
            g.message("t2E = ", t2E)

            if t2E < 0.3:
                t2E_below = t2E
                tau_below = tau
            elif ft0 is not None:
                t2E_above = t2E
                tau_above = tau
                
                lam = (0.3 - t2E_below) / (t2E_above - t2E_below)
                t0 = tau_above * lam + tau_below * (1.0 - lam)

                ainv_sqrt_t0 = t0**0.5
                sqrt_t0_in_OneOverGeV = 0.7292 # this is only an approximation for a quick peek
        
                ainvInGeV = ainv_sqrt_t0 / sqrt_t0_in_OneOverGeV
                ft0.write("%.15g %g" % (t0,ainvInGeV))
                ft0.close()
                ft0 = None

                g.message(f"t0 = {t0}, a^-1 / GeV = {ainvInGeV}")

            if i % ntop == ntop-1 or i == nsteps - 1:
                g.message("Topology")
                Q = g.slice(g.qcd.gauge.topological_charge_5LI(U_wf, cache=c, field=True) / vol3d, 3)
                w.write("Q(%g)" % tau, Q)
                res.append(Q)

                Q = sum(Q).real / len(Q)
                if fQ is not None:
                    fQ.write("%g %.15g\n" % (tau, Q))
                    fQ.flush()

            if fE is not None:
                fE.write("%g %.15g\n" % (tau, E))
                fE.flush()

        g.save(f"{root}/{self.name}/glue", res)
        
    def check(self, root):
        if self.replica != 0:
            return True
        
        config = f"{root}/{self.stream}/ckpoint_lat.{self.tag}"
        n = g.corr_io.count(f"{config}.gluonic")
        g.message("Checking", n)
        # return n in [1207, 2413]
        return True



################################################################################
# Job - Q measurements
################################################################################
class job_measure_Q(job_reproduction_base):
    def __init__(self, stream, conf, name, replica, dependencies):
        self.stream = stream
        self.conf = conf
        self.tag = name
        super().__init__(
            f"{stream}/{conf}_measure_Q",
            ["Q"],
            replica,
            dependencies
        )
        self.weight = 1.0

    def perform_inner(self, root):
        global U, U_mom, css

        config = f"{root}/{self.stream}/ckpoint_lat.{self.tag}"

        eps = 0.01
        nsteps = ensembles[self.stream]["nwf_max"]
        ntop = 200

        U = g.load(config)

        vol3d = float(np.prod(U[0].grid.gdimensions[0:3]))

        if self.replica != 0:
            config = config + "." + str(self.replica)

        if g.rank() == 0:
            fQ = open(f"{config}.Q.alternatives","wt")
        else:
            fQ = None

        res = []
        
        w = g.corr_io.writer(f"{config}.topology")

        action_dbw2 = g.qcd.gauge.action.dbw2(2.0 * U[0].otype.shape[0] / 3.0)
        action_iwasaki = g.qcd.gauge.action.iwasaki(2.0 * U[0].otype.shape[0] / 3.0)

        wilson_flow = g.qcd.gauge.smear.wilson_flow

        def dbw2_flow(U, epsilon, tau):
            return g.qcd.gauge.smear.gradient_flow(U, epsilon, action_dbw2)

        def iwasaki_flow(U, epsilon, tau):
            return g.qcd.gauge.smear.gradient_flow(U, epsilon, action_iwasaki)

        def wilson_iwasaki_dbw2_flow(U, epsilon, tau):
            if tau < 3.0:
                return wilson_flow(U, epsilon)
            elif tau < 6.0:
                return iwasaki_flow(U, epsilon, 0.0)
            else:
                return dbw2_flow(U, epsilon, 0.0)

        def dbw2_iwasaki_wilson_flow(U, epsilon, tau):
            if tau < 3.0:
                return dbw2_flow(U, epsilon, 0.0)
            elif tau < 6.0:
                return iwasaki_flow(U, epsilon, 0.0)
            else:
                return wilson_flow(U, epsilon)

        res = []
        for flow_tag, flow in [
                ("wilson_iwasaki_dbw2", wilson_iwasaki_dbw2_flow),
                ("dbw2_iwasaki_wilson", dbw2_iwasaki_wilson_flow),
                ("dbw2", dbw2_flow),
                ("iwasaki", iwasaki_flow),
        ]:

            U_wf = U
            U_wf = flow(U_wf, epsilon=eps, tau=0.0)
            U_wf = flow(U_wf, epsilon=-eps, tau=0.0)
            g.message(f"Test result {flow_tag}: {(g.norm2(U[0]-U_wf[0])/g.norm2(U[0]))**0.5}")

            tau = 0.0
            U_wf = U
            c = {}
            for i in range(nsteps):
        
                U_wf = flow(U_wf, eps, tau)
                tau += eps
                g.message("%g - %s" % (tau, flow_tag))

                if i % ntop == ntop-1 or i == nsteps - 1:
                    g.message("Topology")
                    Q = g.slice(g.qcd.gauge.topological_charge_5LI(U_wf, cache=c, field=True) / vol3d, 3)
                    w.write("Q(%s,%g)" % (flow_tag,tau), Q)
                    res.append(Q)

                    Q = sum(Q).real / len(Q)
                    if fQ is not None:
                        fQ.write("%s %g %.15g\n" % (flow_tag, tau, Q))
                        fQ.flush()

        g.save(f"{root}/{self.name}/Q", res)
        
    def check(self, root):
        if self.replica != 0:
            return True
        
        config = f"{root}/{self.stream}/ckpoint_lat.{self.tag}"
        n = g.corr_io.count(f"{config}.topology")
        g.message("Checking", n)
        #return n == 1207
        return True




################################################################################
# Create jobs
################################################################################
jobs = []

tags = []

for conf in conf_range:
    for tag in ensembles:
        fn = f"{root_output}/{tag}/ckpoint_lat.{conf}"
        if not os.path.exists(fn):
            g.message(f"Still to do: {fn}")
            tags.append(tag)
        elif (
                os.path.exists(f"{root_output}/{tag}/{conf-1}_checkpoint/verify/.checked") and
                not (
                    os.path.exists(f"{root_output}/{tag}/{conf-1}_measure_Q/verify/.checked") and
                    os.path.exists(f"{root_output}/{tag}/{conf-1}_measure_Q/verify/.checked")
                )
        ):
            g.message(f"Still to do: {fn}")
            tags.append(tag)
    if len(tags) >= len(ensembles):
        break

g.message(f"Current tag priority: {tags}")
for tag in tags:
    latest_conf = None
    g.message(f"Finding latest config for tag {tag}")
    nsteps = ensembles[tag]["nsteps"]
    nsteps_hamiltonian = nsteps // 2
    for conf in conf_range:
        fn = f"{root_output}/{tag}/ckpoint_lat.{conf}"
        if os.path.exists(fn):
            g.message(f"Can start from {fn}")
            if not os.path.exists(f"{root_output}/{tag}/{conf-1}_write_checkpoint_{nsteps-1}"):
                # if we did not write it but it was provided externally, start from here
                latest_conf = conf
                g.message(f"Allowed import {conf}")
            elif (
                    os.path.exists(f"{root_output}/{tag}/{conf-1}_write_checkpoint_{nsteps-1}/verify/.checked")
                    and os.path.exists(f"{root_output}/{tag}/{conf-1}_measure_Q/verify/.checked")
                    and os.path.exists(f"{root_output}/{tag}/{conf-1}_measure_glue/verify/.checked")
            ):
                # if we wrote it, insist that it is verified
                latest_conf = conf
                g.message(f"Allowed complete {conf}")
    if latest_conf is not None:
        g.message(f"ADDING all jobs for {tag} - {latest_conf}; total length={len(jobs)}")
        # first load checkpoint into a state and cleanup non-unitary errors
        job_ckp = [job_checkpoint(tag, latest_conf, r, []) for r in run_replicas]
        job_verify = [job_reproduction_verify(job_ckp)]
        jobs = jobs + job_ckp + job_verify

        # then draw (heatbath)
        job_dr = [job_draw(tag, latest_conf, r, [j.name for j in job_verify]) for r in run_replicas]
        job_verify = [job_reproduction_verify(job_dr)]
        jobs = jobs + job_dr + job_verify + [
            job_release(
                [f"{root_output}/{tag}/{latest_conf}_checkpoint/{r}/config.0" for r in run_replicas]
                + [f"{root_output}/{tag}/{latest_conf}_draw/{r}/state.0" for r in run_replicas[1:]]
                + [f"{root_output}/{tag}/{latest_conf}_draw/{r}/state.draw" for r in run_replicas[1:]],
                job_verify[0]
            )
        ]

        # then do steps of MD integration
        for its in range(nsteps):

            # md step
            job_step = [job_md(tag, latest_conf, its, r, [j.name for j in job_verify]) for r in run_replicas]
            job_verify = [job_reproduction_verify(job_step)]
            jobs = jobs + job_step + job_verify

            # and possibly a Hamiltonian calculation
            if its % nsteps_hamiltonian == nsteps_hamiltonian - 1:
                job_step = [job_hamiltonian(tag, latest_conf, its, r, [j.name for j in job_verify]) for r in run_replicas]
                job_verify = [job_reproduction_verify(job_step)]
                jobs = jobs + job_step + job_verify

            # and release the data
            if its == 0:
                jobs = jobs + [
                    job_release(
                        [f"{root_output}/{tag}/{latest_conf}_draw/0/state.0"]
                        + [f"{root_output}/{tag}/{latest_conf}_md_{its}/{r}/state.0" for r in run_replicas[1:]],
                        job_verify[0]
                    )
                ]
            else:
                jobs = jobs + [
                    job_release(
                        [f"{root_output}/{tag}/{latest_conf}_md_{its-1}/0/state.0"]
                        + [f"{root_output}/{tag}/{latest_conf}_md_{its}/{r}/state.0" for r in run_replicas[1:]],
                        job_verify[0]
                    )
                ]

        # now write checkpoint
        job_step = [job_write_checkpoint(tag, latest_conf, nsteps - 1, latest_conf + 1, r, [j.name for j in job_verify]) for r in run_replicas]
        job_verify = [job_reproduction_verify(job_step)]
        jobs = jobs + job_step + job_verify

        # and release draw and last state
        jobs = jobs + [
            job_release([
                f"{root_output}/{tag}/{latest_conf}_md_{nsteps-1}/{r}/state.0" for r in run_replicas
            ] + [
                f"{root_output}/{tag}/{latest_conf}_draw/0/state.draw"
            ], job_verify[0])
        ]

        # measure glue on latest existing configuration
        job_glue = [job_measure_glue(tag, latest_conf, latest_conf + 1, r, [j.name for j in job_verify]) for r in run_replicas]
        job_verify2 = [job_reproduction_verify(job_glue)]
        jobs = jobs + job_glue + job_verify2

        # measure topology
        job_Q = [job_measure_Q(tag, latest_conf, latest_conf + 1, r, [j.name for j in job_verify]) for r in run_replicas]
        job_verify2 = [job_reproduction_verify(job_Q)]
        jobs = jobs + job_Q + job_verify2




################################################################################
# Execute one job at a time ;  allow for nodefile shuffle outside
################################################################################
for j in jobs:
    g.message(f"Candidate {j.name}")
for i in range(1):
    for ii in range(10):
        g.message("Attempt",ii)
        j=g.jobs.next(root_output, jobs, max_weight=100.0, stale_seconds=3600 * 1.2)
        if j is not None:
            break
        time.sleep(60)
