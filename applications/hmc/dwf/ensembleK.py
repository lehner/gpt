#!/usr/bin/env python3
import gpt as g
import numpy as np
import os, sys, shutil

g.default.set_verbose("defect_correcting_convergence")
g.default.set_verbose("cg_log_convergence")

rng = g.random("test")

# cold start
# target 128x128x128x288
#U = g.qcd.gauge.unit(g.grid([128, 128, 128, 288], g.double)) TODO

U = g.qcd.gauge.random(g.grid([4,4,4,4], g.double), rng)

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

    # check unitarity
    for mu in range(4):
        eps2 = g.norm2(U[mu] * g.adj(U[mu]) - g.identity(U[mu])) / g.norm2(U[mu])
        g.message("Unitarity defect for mu=",mu,"is",eps2)

        eps2 = g.norm2(g.matrix.det(U[mu]) - g.identity(g.complex(U[0].grid))) / U[0].grid.gsites
        g.message("Determinant defect for mu=",mu,"is",eps2)

        g.message("Project")
        g.project(U[mu], "defect_left")

        eps2 = g.norm2(U[mu] * g.adj(U[mu]) - g.identity(U[mu])) / g.norm2(U[mu])
        g.message("Unitarity defect for mu=",mu,"is",eps2)

        eps2 = g.norm2(g.matrix.det(U[mu]) - g.identity(g.complex(U[0].grid))) / U[0].grid.gsites
        g.message("Determinant defect for mu=",mu,"is",eps2)
else:
    latest_it = -1

if False:
    Lold = U[0].grid.gdimensions
    Lnew = [2*Lold[0],2*Lold[1],2*Lold[2]] + [3*Lold[3]]
    grid = g.grid(Lnew, g.double)
    Unew = []
    for u in U:
        unew = g.mcolor(grid)
        for n0 in range(2):
            for n1 in range(2):
                for n2 in range(2):
                    for n3 in range(3):
                        g.message(n0,n1,n2,n3)
                        if grid.processor == 0:
                            unew[
                                n0*Lold[0]:(n0+1)*Lold[0],
                                n1*Lold[1]:(n1+1)*Lold[1],
                                n2*Lold[2]:(n2+1)*Lold[2],
                                n3*Lold[3]:(n3+1)*Lold[3]
                            ] = u[
                                0:Lold[0],
                                0:Lold[1],
                                0:Lold[2],
                                0:Lold[3]
                            ]
                        else:
                            unew[
                                0:0,
                                0:0,
                                0:0,
                                0:0
                            ] = u[
                                0:0,
                                0:0,
                                0:0,
                                0:0
                            ]
        Unew.append(unew)
    U = Unew
    
    g.message("New", g.qcd.gauge.plaquette(U))
    g.save(f"{dst}/ckpoint_lat.{it0}", U, g.format.nersc())
    sys.exit(0)

ckp = g.checkpointer(f"{dst}/checkpoint2")
ckp.grid = U[0].grid

pc = g.qcd.fermion.preconditioner
inv = g.algorithms.inverter
eofa_ratio = g.qcd.pseudofermion.action.exact_one_flavor_ratio

g.message("Plaquette",g.qcd.gauge.plaquette(U))

# FT
rho = 0.12
sm = [
    g.qcd.gauge.smear.local_stout(rho=rho, dimension=mu, checkerboard=p) for mu in range(4) for p in [g.even, g.odd]
]

# undo the smearing
U0 = g.copy(U)
for s in sm:
    U = s.inv(U)
U1 = g.copy(U)
for s in reversed(sm):
    U1 = s(U1)

for mu in range(4):
    g.message(f"Test s(s.inv) = id, U_{mu} = {g.norm2(U1[mu] - U0[mu]) / g.norm2(U0[mu])}")

g.message("Plaquette of integration field",g.qcd.gauge.plaquette(U))


def two_flavor_ratio(fermion, m1, m2, solver):
    M1 = fermion(m1, m1)
    M2 = fermion(m2, m2)
    return g.qcd.pseudofermion.action.two_flavor_ratio_evenodd_schur([M1, M2], solver)


def light(U0, m_plus, m_minus):
    return g.qcd.fermion.mobius(
        U0,
        mass_plus=m_plus,
        mass_minus=m_minus,
        M5=1.8,
        b=1.5,
        c=0.5,
        Ls=12,
        boundary_phases=[1, 1, 1, -1],
    )




pc = g.qcd.fermion.preconditioner
inv = g.algorithms.inverter
sympl = g.algorithms.integrator.symplectic

lq = light(U, 1, 1)
F_grid_eo = lq.F_grid_eo
F_grid = lq.F_grid
lq = None

sloppy_prec = 1e-9
sloppy_prec_light = 1e-9
exact_prec = 1e-11

cg_s_inner = inv.cg({"eps": 1e-4, "eps_abs": sloppy_prec * 0.15, "maxiter": 40000, "miniter": 50})
cg_s_light_inner = inv.cg({"eps": 1e-4, "eps_abs": sloppy_prec_light * 0.15, "maxiter": 40000, "miniter": 50})
cg_e_inner = inv.cg({"eps": 1e-4, "eps_abs": exact_prec * 0.3, "maxiter": 40000, "miniter": 50})

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


# conjugate momenta
U_mom = g.group.cartesian(U)

action_gauge_mom = g.qcd.scalar.action.mass_term()
action_gauge = g.qcd.gauge.action.iwasaki(2.44)  # changed from 2.41 at traj=295

# first estimate based on crude E extrapolation, likely 4% off for m_l and m_s:
#( (0.0003546 + 0.0176)/27.34 - 0.0003546) = 0.00030211543525969275

# new estimation based on global fit should do better:
# m_s = 0.01682836
# m_l = 0.00028884

if latest_it <= 603:
    g.message("Use first-generation masses")
    m_l = 0.000302
    m_s = 0.0176

else:
    g.message("Use second-generation masses")

    # global fit estimate (mk)
    # m_l = 0.000289
    # m_s = 0.0168

    # simple estimate via m_SS msw0Zh dependence ; then use 27.2 as quark mass ratio incl. m_res
    # m_l = 0.000282
    # m_s = 0.0170   (another estimate mk5 agrees with this)

    m_l = 0.000286
    m_s = 0.0169



rat = g.algorithms.rational.zolotarev_inverse_square_root(1.0**0.5, 70.0**0.5, 11)
# before 11 highest, led to 1e-9 error, 6 led to 1.2435650287301314e-08,
# 13 led to 8e-10, 20 to 1.320967779605553e-10, 3 to 7.120046807695957e-08
# ran power method below, seems like 70 is best upper level
rat_fnc = g.algorithms.rational.rational_function(rat.zeros, rat.poles, rat.norm)

# see params.py for parameter motivation
css = [
    ([], 8, F_grid_eo),
    ([], 16, F_grid_eo),
    ([], 4, F_grid)
]

def store_css():
    global css
    ckp_css = g.checkpointer(f"{dst}/checkpoint3")
    ckp_css.grid = U[0].grid

    for cc, ncc, cgrid in css:
        ckp_css.save([float(len(cc))])
        for i in range(len(cc)):
            g.message("Saving field norm2", g.norm2(cc[i]))
            g.message("with grid", str(cc[i].grid))
            ckp_css.save(cc[i])

def load_css():
    global css

    if os.path.exists(f"{dst}/checkpoint3"):
        ckp_css = g.checkpointer(f"{dst}/checkpoint3")
        ckp_css.grid = U[0].grid

        for cc, ncc, cgrid in css:
            params = [0.0]
            if not ckp_css.load(params):
                g.message("No more fields to load")
                break
            g.message(f"Loading {int(params[0])} solution fields")
            for i in range(int(params[0])):
                nn = g.vspincolor(cgrid)
                if ckp_css.load(nn):
                    cc.append(nn)

def store_cfields(tag, flds):
    ckp_css = g.checkpointer(f"{dst}/checkpoint.{tag}")
    ckp_css.grid = U[0].grid

    for i in range(len(flds)):
        ckp_css.save(flds[i])

def load_cfields(tag, flds):
    if os.path.exists(f"{dst}/checkpoint.{tag}"):
        ckp_css = g.checkpointer(f"{dst}/checkpoint.{tag}")
        ckp_css.grid = U[0].grid

        if not ckp_css.load(flds):
            return False
        g.message(f"Successfully restored {tag}")
        return True
    return False



load_css()

hasenbusch_ratios = [  # Nf=2+1
    (0.65, 1.0, None, two_flavor_ratio, mk_chron(cg_e, *css[0]), mk_chron(cg_s, *css[0]), light),
    (0.28, 0.65, None, two_flavor_ratio, mk_chron(cg_e, *css[0]), mk_chron(cg_s, *css[0]), light),
    (0.11, 0.28, None, two_flavor_ratio, mk_chron(cg_e, *css[0]), mk_chron(cg_s, *css[0]), light),
    (0.017, 0.11, None, two_flavor_ratio, mk_chron(cg_e, *css[0]), mk_chron(cg_s, *css[0]), light),
    (0.004, 0.017, None, two_flavor_ratio, mk_chron(cg_e, *css[0]), mk_chron(cg_s, *css[0]), light),
    (0.0019, 0.004, None, two_flavor_ratio, mk_chron(cg_e, *css[1]), mk_chron(cg_s, *css[1]), light),
    (m_l, 0.0019, None, two_flavor_ratio, mk_chron(cg_e, *css[1]), mk_chron(cg_s_light, *css[1]), light),
    (0.23, 1.0, rat_fnc, eofa_ratio, mk_slv_e(*css[2]), mk_slv_s(*css[2]), light),
    (m_s, 0.23, rat_fnc, eofa_ratio, mk_slv_e(*css[2]), mk_slv_s(*css[2]), light),
]

fields = [
    (U + [g.vspincolor(F_grid_eo)]),
    (U + [g.vspincolor(F_grid_eo)]),
    (U + [g.vspincolor(F_grid_eo)]),
    (U + [g.vspincolor(F_grid_eo)]),
    (U + [g.vspincolor(F_grid_eo)]),
    (U + [g.vspincolor(F_grid_eo)]),
    (U + [g.vspincolor(F_grid_eo)]),
    (U + [g.vspincolor(U[0].grid)]),
    (U + [g.vspincolor(U[0].grid)]),
]

# exact actions
action_fermions_e = [
    af(lambda m_plus, m_minus, qqi=qq: qqi(U, m_plus, m_minus), m1, m2, se)
    for m1, m2, rf, af, se, ss, qq in hasenbusch_ratios
]

# sloppy actions
action_fermions_s = [
    af(lambda m_plus, m_minus, qqi=qq: qqi(U, m_plus, m_minus), m1, m2, ss)
    for m1, m2, rf, af, se, ss, qq in hasenbusch_ratios
]


if False:
    g.default.push_verbose("power_iteration_convergence", True)
    sr = g.algorithms.eigen.power_iteration(eps=1e-5, real=True, maxiter=40)(
        action_fermions_s[-1].matrix(fields[-1]), rng.normal(fields[-1][-1])
    )
    g.default.pop_verbose()
    g.message("Spectral range", sr)


metro = g.algorithms.markov.metropolis(rng)

pure_gauge = True

def transform(aa, s, i):
    aa_transformed = aa[i].transformed(s, indices=list(range(len(U))))
    aa_orig = aa[i]
    def draw(fields, rng, *extra):
        sfields = s(fields[0:len(U)]) + fields[len(U):]
        x = aa_orig.draw(sfields, rng, *extra)
        return x
    aa_transformed.draw = draw
    aa[i] = aa_transformed
    
a_log_det = None
for s in sm:
    action_gauge = action_gauge.transformed(s)
    for i in range(len(hasenbusch_ratios)):
        transform(action_fermions_e, s, i)
        transform(action_fermions_s, s, i)
    if a_log_det is None:
        a_log_det = s.action_log_det_jacobian()
    else:
        a_log_det = a_log_det.transformed(s) + s.action_log_det_jacobian()

def hamiltonian(draw):
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

iq = sympl.update_q(
    U, log(lambda: action_gauge_mom.gradient(U_mom, U_mom), "gauge_mom")
)

ip_gauge = sympl.update_p(U_mom, gauge_force)
ip_fermion = sympl.update_p(U_mom, fermion_force)
ip_log_det = sympl.update_p(U_mom, log_det_force)
ip_log_det_sp = sympl.update_p(U_mom, log_det_force_sp)

ip_gauge_fg = sympl.update_p_force_gradient(U, iq, U_mom, ip_gauge, ip_gauge)
ip_fermion_fg = sympl.update_p_force_gradient(U, iq, U_mom, ip_fermion, ip_fermion)
ip_log_det_fg = sympl.update_p_force_gradient(U, iq, U_mom, ip_log_det, ip_log_det)#_sp

mdint = sympl.OMF2_force_gradient(
    2, ip_fermion,
    sympl.OMF2_force_gradient(2, ip_log_det,
                              sympl.OMF2_force_gradient(2, ip_gauge, iq, ip_gauge_fg),
                              ip_log_det_fg),
    ip_fermion_fg
)


no_accept_reject = True


nsteps = 40
tau = 8.0

tau = 12.0
nsteps = 4 # TODO: modify

def refresh_momentum():
    global ckp
    params = U_mom + [x[-1] for x in fields] + [0.0, 0.0]
    if not ckp.load(params):
        h0, s0 = hamiltonian(True)
        params[-2] = h0
        params[-1] = s0
        ckp.save(params)
        store_css()
        g.barrier()
        sys.exit(0)
    else:
        h0, s0 = params[-2:]
    return h0, s0

def evolve(tau_local, nsteps_local):
    global ckp

    if not ckp.load(U_mom + U):
        its0 = nsteps_local - 1
        while its0 >= 0:
            g.message(f"Try to load fields after iteration {its0}")
            if load_cfields(f"{its0}", U_mom + U):
                break
            its0 -= 1
        its0 += 1
        for its in range(its0, nsteps_local):
            g.message(f"tau-iteration: {its} -> {tau/nsteps_local*its}")
            mdint(tau_local / nsteps_local)
            store_css()
            store_cfields(f"{its}", U_mom + U)

            if g.rank() == 0:
                flog = open(f"{dst}/current.log.{its}","wt")
                for x in log.grad:
                    flog.write(f"{x} force norm2/sites = {np.mean(log.get(x))} +- {np.std(log.get(x))}\n")
                flog.write(f"Timing:\n{log.time}\n")
                flog.close()

            if its < nsteps_local - 1:
                g.barrier()
                sys.exit(0)
                
        ckp.save(U_mom + U)

        g.barrier()

        # reset checkpoint
        g.message("Reset evolve checkpoints")
        if g.rank() == 0:
            for it in range(nsteps_local):
                if os.path.exists(f"{dst}/checkpoint.{it}"):
                    shutil.rmtree(f"{dst}/checkpoint.{it}")

        g.barrier()


def metropolis_evolve(tau_local, nsteps_local, h0, s0, status):
    global ckp
    
    accrej = metro(U_mom + U)

    evolve(tau_local, nsteps_local)

    params = [0.0, 0.0]
    if not ckp.load(params):
        h1, s1 = hamiltonian(False)
        g.message(f"dH = {h1-h0}")
        ckp.save([h1,s1])
        sys.exit(0)        
    else:
        h1, s1 = params
        
    g.message(f"dH = {h1-h0}")
    status.append(h1-h0)
    status.append(s1-s0)
    
    if not no_accept_reject:
        if not accrej(h1, h0):
            # restore to zero state
            g.message("Reject update with dH=",h1-h0)
            # flip momenta
            for um in U_mom:
                um *= -1
            h1, s1 = h0, s0
            acc = False
        else:
            g.message("Accept update with dH=",h1-h0)
            acc = True
    else:
        acc = True

    status.append(acc)

    g.message(f"Plaquette after evolution with accept/reject is {g.qcd.gauge.plaquette(U)}")
    
    return h1, s1, acc
    

def hmc(tau):
    global ff_iterator, ckp
    ff_iterator = 0
    g.message("After metro")

    h0, s0 = refresh_momentum()
    
    g.message("After H(true)",h0,s0)

    status = []
    h1, s1, acc1 = metropolis_evolve(tau/2, nsteps//2, h0, s0, status)

    g.message("After first evolution")
    h2, s2, acc2 = metropolis_evolve(tau/2, nsteps//2, h1, s1, status)

    g.message("After second evolution")
    
    store_css()

    return status
    
accept, total = 0, 0
for it in range(it0, N):
    pure_gauge = it < 100
    no_accept_reject = it < 1 # TODO modify
    g.message(pure_gauge, no_accept_reject)

    dH_1, dS_1, acc_1, dH_2, dS_2, acc_2 = hmc(tau)

    Uft = U
    for s in reversed(sm):
        Uft = s(Uft)
        
    plaq = g.qcd.gauge.plaquette(U)
    plaqft = g.qcd.gauge.plaquette(Uft)
    g.message(f"HMC {it} has P = {plaqft}, Pft = {plaq}, dS_1 = {dS_1}, dS_2 = {dS_2}, dH_1 = {dH_1}, dH_2 = {dH_2}, acceptance = {acc_1} {acc_2}")
    #for x in log.grad:
    #    g.message(f"{x} force norm2/sites =", np.mean(log.get(x)), "+-", np.std(log.get(x)))
    #g.message(f"Timing:\n{log.time}")

    if g.rank() == 0:
        flog = open(f"{dst}/ckpoint_lat.{it}.log","wt")
        flog.write(f"dH_1 {dH_1}\n")
        flog.write(f"dH_2 {dH_2}\n")
        flog.write(f"accept_1 {acc_1}\n")
        flog.write(f"accept_2 {acc_2}\n")
        flog.write(f"P {plaqft}\n")
        flog.write(f"Pft {plaq}\n")
        flog.close()

    if it % 10 == 0:
        # reset statistics
        log.reset()
        g.message("Reset log")

    g.save(f"{dst}/config.{it}", Uft)
    
    g.barrier()

    g.save(f"{dst}/ckpoint_lat.{it}", Uft, g.format.nersc())

    g.barrier()

    # reset checkpoint
    if g.rank() == 0:
        shutil.rmtree(f"{dst}/checkpoint2")
    
    #rng = g.random(f"new{dst}-{it}", "vectorized_ranlux24_24_64")

    g.barrier()
    break
