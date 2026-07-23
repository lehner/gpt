#!/usr/bin/env python3
import gpt as g
import numpy as np

pc = g.qcd.fermion.preconditioner
inv = g.algorithms.inverter

config = g.default.get("--config", None)
phase = g.default.get("--phase", None)
t0 = g.default.get_int("--t0", None)
assert phase in ["mres", "final"]

# measured in phase "mres"
mres_db = {
(1,12) : 0.000176987243652344,
(1,14) : 7.53514709472658e-05,
(1,16) : 3.32426910400391e-05,
(1,20) : 7.08174133300782e-06,
(1.25,12) : 5.54824829101563e-05,
(1.25,14) : 1.88056945800781e-05,
(1.25,16) : 6.89265441894532e-06,
(1.25,20) : 1.12191772460939e-06,
(1.5,12) : 8.24175720214845e-05,
(1.5,14) : 1.96361312866211e-05,
(1.5,16) : 4.78878021240235e-06,
(1.5,20) : 3.65005493164074e-07,
}

U = g.load(config)
T = U[0].grid.gdimensions[3]

src0 = g.mspincolor(U[0].grid)
src1 = g.mspincolor(U[0].grid)
rng = g.random(f"{config}.mres.{t0}")
t1 = (t0 + T // 2) % T
g.create.wall.z2(src0, t0, rng)
g.create.wall.z2(src1, t1, rng)

w = g.corr_io.writer(f"{config}.mres-{phase}.{t0}")

exact_prec = 1e-12
cg_e_inner = inv.cg({"eps": 1e-4, "eps_abs": exact_prec * 0.3, "maxiter": 40000, "miniter": 50, "fail_if_not_converged": True})
cg_e_inner = inv.preconditioned(pc.eo2_ne(), cg_e_inner)
slv = inv.defect_correcting(
    inv.mixed_precision(cg_e_inner, g.single, g.double),
    eps=exact_prec,
    maxiter=100,
)

for alpha in [1, 1.25, 1.5]:
    # b+c = alpha
    # b-c = 1
    b = (alpha + 1) / 2
    c = (alpha - 1) / 2
    m_mres = 0.0292 # mass to calculate mres
    M5 = 1.8

    for Ls in [12, 14, 16, 20]:
        tag = f"{alpha}.{Ls}"

        if phase == "mres":
            mres = None
        else:
            mres = mres_db[(alpha, Ls)]

        if phase == "mres":
            assert mres is None
            
            F = g.qcd.fermion.mobius(
                U,
                mass=m_mres,
                M5=M5,
                b=b,
                c=c,
                Ls=Ls,
                boundary_phases=[1, 1, 1, -1],
            )

            prop_5d_s = g(F.bulk_propagator(slv) * src0)
            prop_s = g(F.bulk_propagator_to_propagator * prop_5d_s)

            corr = g.slice(g.trace(g.adj(prop_s) * prop_s), 3)
            corr = corr[t0:] + corr[:t0]
            w.write(f"{tag}/ss.G5G5", corr)
            
            p = F.J5q(prop_5d_s)
            corr5 = g.slice(g.trace(p * g.adj(p)), 3)
            corr5 = corr5[t0:] + corr5[:t0]
            w.write(f"{tag}/mres.ss.G5G5", corr5)

        elif phase == "final":
            assert mres is not None
            w.write(f"{tag}/mres", [mres])
            
            F = g.qcd.fermion.mobius(
                U,
                mass=m_mres - mres,
                M5=M5,
                b=b,
                c=c,
                Ls=Ls,
                boundary_phases=[1, 1, 1, -1],
            )
        
            prop_x0 = g(F.propagator(slv) * src0)
            prop_x1 = g(F.propagator(slv) * src1)
            prop_0x = g( g.gamma[5] * g.adj(prop_x0) * g.gamma[5] )
            prop_10 = g(g.adj(src1) * prop_x0)
            prop_10 = g.mspincolor(src1.grid.globalsum(np.sum(prop_10[:], axis=0)))

            # vv correlator
            for i in range(3):
                corr = g.slice(g.trace(g.gamma[i] * prop_0x * g.gamma[i] * prop_x0), 3)
                corr = corr[t0:] + corr[:t0]
                w.write(f"{tag}/ss(-mres).G{i}G{i}", corr)

            # now compute zv
            corr_2pt = [g.trace(g.adj(prop_10) * prop_10).real]
            corr_3pt = g.slice(g.trace(g.gamma[5].tensor() * prop_10 * g.gamma[5].tensor() * prop_0x * g.gamma[3] * prop_x1), 3)
            corr_3pt = corr_3pt[t0:] + corr_3pt[:t0]

            w.write(f"{tag}/zv.3pt(-mres)", np.array(corr_3pt))
            w.write(f"{tag}/zv.2pt(-mres)", np.array(corr_2pt))
    
del w
g.barrier()
