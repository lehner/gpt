#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2025
#
import gpt as g
import numpy as np
import sys, os

# setup rng, mute
g.default.set_verbose("random", True)
g.default.set_verbose("random_performance", True)
rng = g.random("test_mg", "vectorized_ranlux24_24_64")

# setup gauge field
U = g.load("48I.1000") # 24ID

# default grid
grid = U[0].grid

# create source
src = g.vspincolor(grid)
src[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])

# abbreviations
i = g.algorithms.inverter
p = g.qcd.fermion.preconditioner

# define transitions between grids (setup)
def find_near_null_vectors(w, cgrid):
    if os.path.exists("basis2.48I.60"):
        basis = g.load("basis2.48I.60")
        bm = g.block.map(cgrid, basis)
    else:
        start = w.vector_space[0].lattice()
        rng.cnormal(start)
        cop = g.algorithms.polynomial.chebyshev(low=4e-5, high=5, order=300)
        irl = g.algorithms.eigen.irl(Nk=60, Nm=100, Nstop=60, resid=1e-8, betastp=1e-5, maxiter=40, Nminres=1)
        basis, evals = irl(cop(w), start)

        # test near null vectors
        for b in basis:
            lambda_eff = (g.norm2(w*b) / g.norm2(b))**0.5
            lambda_eff2 = g.inner_product(b, w*b) / g.norm2(b)
            g.message(f"Effective EV for near null vector: {lambda_eff} {lambda_eff2}")

        bm = g.block.map(cgrid, basis)
        for _ in range(2):
            bm.orthonormalize()
        bm.check_orthogonality()
        g.save("basis2.48I.60", basis)
        sys.exit(0)
    return bm


##############
# DWF HDCG illustration
dwf_dp = g.qcd.fermion.mobius(U,
    mass=0.00078,
    M5=1.8,
    b=1.5,
    c=0.5,
    Ls=24,
    boundary_phases=[1, 1, 1, -1],
)
dwf_sp = dwf_dp.converted(g.single)

g.default.set_verbose("cg", True)
g.default.set_verbose("chebyshev", True)
g.default.push_verbose("defect_correcting_convergence", True)
#g.default.pop_verbose()

block_cg = i.block_cg({"eps": 1e-5, "maxiter": 100})

g.default.push_verbose("fgmres", True)
g.default.push_verbose("fgmres_convergence", True)
coarse_cg = i.cg({"eps": 1e-6, "maxiter": 5000})

g.message("Find nnsv")
bm = find_near_null_vectors(p.eo2_ne()(dwf_sp).Mpc, g.block.grid(dwf_sp.F_grid_eo, [24,4,4,4,4]))



g.mem_report(details=False)

g.message("Coarsen")
def coarsen_operator_hdcg(bm, op):
    if os.path.exists("points2.48I.60"):
        pnts = g.load("points2.48I.60")
    else:
        cop = bm.coarse_operator(op).compile(max_point_norm=4, max_point_per_dimension=[0,1,1,1,1], tolerance=None, nblock=2)
        g.save("points2.48I.60", cop.points)
        sys.exit(0)

    rcop = g.block.matrix_operator.compiled({eval(x): pnts[x] for x in pnts}, implementation="reference")
    bcop = g.block.matrix_operator.compiled({eval(x): pnts[x] for x in pnts}, implementation="blas")
    scop = g.block.matrix_operator.compiled({eval(x): pnts[x] for x in pnts}, implementation="stencil")
    
    # test hermiticity
    vecs = rng.cnormal([op.vector_space[1].lattice() for _ in range(2)])
    g.message(abs(g.inner_product(vecs[0], op * vecs[1])/g.inner_product(vecs[1], op * vecs[0]).conjugate() - 1), "fine")

    vecs = bm.project(vecs)
    g.message(abs(g.inner_product(vecs[0], rcop * vecs[1])/g.inner_product(vecs[1], rcop * vecs[0]).conjugate() - 1), "reference coarse")
    g.message(abs(g.inner_product(vecs[0], bcop * vecs[1])/g.inner_product(vecs[1], bcop * vecs[0]).conjugate() - 1), "blas coarse")
    g.message(abs(g.inner_product(vecs[0], scop * vecs[1])/g.inner_product(vecs[1], scop * vecs[0]).conjugate() - 1), "stencil coarse")

    x = g(rcop * vecs[1])
    y = g(bcop * vecs[1])
    g.message("Error of blas/reference:",(g.norm2(x-y)/g.norm2(x))**0.5)

    scop = bm.coarse_operator(op)
    x = g(scop * vecs[1])
    g.message("Error of blas/projected:",(g.norm2(x-y)/g.norm2(x))**0.5)

    g.message(abs(g.inner_product(vecs[0], scop * vecs[1])/g.inner_product(vecs[1], scop * vecs[0]).conjugate() - 1), "slow coarse")

    return bcop

if False:
    x=rng.cnormal(g.vspincolor(dwf_sp.F_grid_eo))
    x.checkerboard(g.odd)

    MM = p.eo2_ne()(dwf_sp).Mpc
    g(g.algorithms.inverter.cg(eps=1e-8, maxiter=40000)(MM) * x)

    g(g.algorithms.inverter.fgmres(eps=1e-8, maxiter=40000, restartlen=15)(MM) * x)
    sys.exit(0)

if False:
    x=rng.cnormal(g.vspincolor(dwf_sp.F_grid_eo))
    x.checkerboard(g.odd)

    MM = p.eo2_ne()(dwf_sp).Mpc # 4.6403594657997225 upper edge
    irl = g.algorithms.eigen.irl(Nk=96, Nm=128, Nstop=96, resid=1e-8, betastp=1e-5, maxiter=40, Nminres=1)
    evals, _ = irl(MM, x)
    g.message(evals)

    MM = p.eo3_ne()(dwf_sp).Mpc # 89.80516975639685 upper edge
    irl = g.algorithms.eigen.irl(Nk=96, Nm=128, Nstop=96, resid=1e-8, betastp=1e-5, maxiter=40, Nminres=1)
    evals, _ = irl(MM, x)
    g.message(evals)

if False:
    x=rng.cnormal(g.vspincolor(dwf_sp.F_grid_eo))
    x.checkerboard(g.odd)

    MM = p.eo2_ne()(dwf_sp).Mpc # 4.6403594657997225 - x == 4.640357047653548 -> 2.4e-6 lower edge
    irl = g.algorithms.eigen.irl(Nk=96, Nm=128, Nstop=96, resid=1e-8, betastp=1e-5, maxiter=40, Nminres=1)
    evals, _ = irl(lambda dst, src: g(dst, src*4.6403594657997225 - MM*src), x)
    g.message(evals)

    MM = p.eo3_ne()(dwf_sp).Mpc # 89.80516975639685 - x == 89.80512335986033 -> 4.6e-5 lower edge
    irl = g.algorithms.eigen.irl(Nk=96, Nm=128, Nstop=96, resid=1e-8, betastp=1e-5, maxiter=40, Nminres=1)
    evals, _ = irl(lambda dst, src: g(dst, src*89.80516975639685 - MM*src), x)
    g.message(evals)

    # Just to document this here (probably not a good idea since skype is going away soon):
    # your preconditioner in the HDCG has spectral range of 4.6e-5 .. 90 on a configuration where my previous
    # one has 2.4e-6 .. 4.6 .  The condition numbers are 2.0e6 and 1.9e6 respectively.
    # So indeed it does not matter apart from the adjustment of the parameters.

    # need to scale Peter's parameters by 1/20
    
    sys.exit(0)
    
MM = p.eo2_ne()(dwf_sp).Mpc
cop = coarsen_operator_hdcg(bm, MM)

if not os.path.exists("coarse_deflate2.48I.60"):
    # cop spectrum to 1.86
    # MM to 4.6
    c = g.algorithms.polynomial.chebyshev({"low": 1e-3, "high": 2.5, "order": 40})

    irl = g.algorithms.eigen.irl(Nk=192, Nm=256, Nstop=192, resid=1e-8, betastp=1e-5, maxiter=40, Nminres=1)
    x=rng.cnormal(g.vspincolor(dwf_sp.F_grid_eo))
    x.checkerboard(g.odd)
    v = bm.project(x)

    #g.message(g.algorithms.eigen.power_iteration(eps=1e-3,maxiter=50)(cop, v))
    #sys.exit(0)
    
    evec, _ = irl(c(cop), v)
    evals, eps2 = g.algorithms.eigen.evals(cop, evec, real=True)
    for i in range(len(evec)):
        g.message(i, evals[i], eps2[i])
        evc = bm.promote(evec[i])
        evals2, eps22 = g.algorithms.eigen.evals(MM, [evc], real=True)
        g.message(evals2, eps22)
    g.save("coarse_deflate2.48I.60", (evec, evals))
    sys.exit(0)

evec, evals = g.load("coarse_deflate2.48I.60")

if False:
    for i in range(len(evec)):
        evc = bm.promote(evec[i])
        evals2, eps22 = g.algorithms.eigen.evals(MM, [evc], real=True)
        g.message(i, evals[i], evals2, eps22)
    sys.exit(0)

if False:
    x=rng.cnormal(g.vspincolor(dwf_sp.F_grid_eo))
    x.checkerboard(g.odd)
    x /= g.norm2(x) ** 0.5
    v = bm.project(x)
    MM = p.eo2_ne()(dwf_sp).Mpc
    cop = coarsen_operator_hdcg(bm, MM)
    g.message(g.algorithms.eigen.power_iteration(eps=1e-3,maxiter=50)(cop, v))
    sys.exit(0)


# prepare solve tests
MM = p.eo2_ne()(dwf_dp).Mpc

x=rng.cnormal(g.vspincolor(dwf_dp.F_grid_eo))
x.checkerboard(g.odd)
x /= g.norm2(x) ** 0.5


if True:
    g.message("coarse_deflate, fgmres")
    g.default.set_verbose("cg_convergence", False)

    xxx = i.sequence(
        g.algorithms.polynomial.chebyshev(low=2.0/20.0, high=92.2 / 20, order=8, func=lambda x: 1/x),
        i.relaxation(i.coarse_grid(
            i.sequence(
                i.deflate(evec, evals),
                #i.relaxation(g.algorithms.polynomial.chebyshev(low=0.02/20,high=40/20,order=120, func=lambda x: 1/x))
                i.chebyshev(low=0.00019,high=2.1,maxiter=120,eps=1e-15)
            ),
            bm, coarsen_operator_hdcg
        )),
    )
    g.default.set_verbose("cg_convergence", True)
    hdcg_inner = i.fgmres(
        eps=1e-8, maxiter=1000, restartlen=30,
    #hdcg_inner = i.cg(#restartlen=3,
    #    eps=1e-8, maxiter=1000,
        prec=i.mixed_precision(
            xxx,
            g.single,
            g.double
        ),
    )

    # 124 outer iterations with FGMRES and new setup -> 1116 fine-grid multiplies
    hdcg_inner(MM)(x)
    
sys.exit(0)



# slv_hdcg = 
slv_5d = i.preconditioned(
    p.eo2_ne(),
    hdcg_inner
)

prop = dwf_dp.propagator(slv_5d)
g(prop * src)


g.message(f"HDCG lowered iteration number from {len(slv_smoother_only.history)} to {len(slv_hdcg.history)}")
