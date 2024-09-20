#!/usr/bin/env python3
# Idea: test communication for correctness in arbitrary setups
import gpt as g

arg_grid = g.default.get_ivec("--grid", None, 4)

assert arg_grid is not None


g.default.set_verbose("random", False)
rng = g.random(
    "test", "vectorized_ranlux24_24_64"
)

def D_DWF(dst, src, U, b, c, mass, M5):
    src_s = g.separate(src, 0)
    dst_s = [g.lattice(s) for s in src_s]

    D_W = g.qcd.fermion.reference.wilson_clover(U, mass=-M5, csw_r=0.0, csw_t=0.0, nu=1.0, xi_0=1.0,
                                                isAnisotropic=False,
                                                boundary_phases=[1,1,1,-1])

    Ls = len(src_s)
    
    src_plus_s = []
    src_minus_s = []
    for s in range(Ls):
        src_plus_s.append(g(0.5 * src_s[s] + 0.5 * g.gamma[5]*src_s[s]))
        src_minus_s.append(g(0.5 * src_s[s] - 0.5 * g.gamma[5]*src_s[s]))
    for d in dst_s:
        d[:] = 0
    for s in range(Ls):
        dst_s[s] += b*D_W* src_s[s] + src_s[s]
    for s in range(1,Ls):
        dst_s[s] += c*D_W * src_plus_s[s-1] - src_plus_s[s-1]
    for s in range(0,Ls-1):
        dst_s[s] += c*D_W * src_minus_s[s+1] - src_minus_s[s+1]
    dst_s[0] -= mass*(c*D_W * src_plus_s[Ls-1] - src_plus_s[Ls-1])
    dst_s[Ls-1] -= mass*(c*D_W * src_minus_s[0] - src_minus_s[0])
            
    dst @= g.merge(dst_s, 0)

for precision in [g.single, g.double]:
    grid = g.grid(arg_grid, precision)

    # test mobius dwf against cshift version
    U = g.qcd.gauge.random(grid, rng)
    Ls = 12
    b = 1.5
    c = 0.5
    M5 = 1.8
    mass = 0.123
    mobius = g.qcd.fermion.mobius(
        U,
        Ls=Ls,
        mass=mass,
        b=b,
        c=c,
        M5=M5,
        boundary_phases=[1,1,1,-1]
    )

    src = rng.cnormal(g.vspincolor(mobius.F_grid))
    dst = g(mobius * src)

    dst_ref = g.lattice(dst)
    dst_ref[:] = 0
    D_DWF(dst_ref, src, U, b, c, mass, M5)

    eps = (g.norm2(dst_ref - dst) / g.norm2(dst_ref)) ** 0.5
    g.message(f"Test mobius implementation: {eps}")
    assert eps < precision.eps * 100

    # test global sums
    for it in range(10):
        local_value = g.random(str(it) + "-" + str(grid.processor)).cnormal()
        global_value = grid.globalsum(local_value)

        ref_global_value = 0.0
        for i in range(grid.Nprocessors):
            ref_global_value += g.random(str(it) + "-" + str(i)).cnormal()

        eps = abs(global_value - ref_global_value) / abs(global_value)
        g.message(f"Test global sum {i}: {eps}")
        assert eps < precision.eps * 100

# now have double-precision grid
exact_prec = 1e-10
pc = g.qcd.fermion.preconditioner
inv = g.algorithms.inverter
g.default.set_verbose("cg_convergence")
g.default.set_verbose("mixed_precision_performance")
cg_e_inner = inv.cg({"eps": 1e-4, "eps_abs": exact_prec * 0.3, "maxiter": 40000, "miniter": 50})
cg_e = inv.defect_correcting(
    inv.mixed_precision(inv.preconditioned(pc.eo2_ne(), cg_e_inner), g.single, g.double),
    # inv.preconditioned(pc.eo2_ne(), cg_e_inner),
    eps=exact_prec,
    maxiter=100,
)

prop = cg_e(mobius)
dst0 = g(prop * src)
dst1 = g(prop * src)
eps = (g.norm2(dst0 - dst1)/g.norm2(dst0))**0.5
g.message(f"Reproduce mixed precision solve: {eps}")
assert eps < precision.eps * 100
