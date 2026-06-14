#!/usr/bin/env python3
import gpt as g
import numpy as np

g.default.set_verbose("blas", True)
g.default.set_verbose("contract_plan", True)

n = g.default.get_int("--n", 16**4)

g.message("Using n =",n)

tmp = g.accelerator_buffer_manager()
target = g.accelerator_buffer(shape=(n,), dtype=np.complex128)
prop = g.accelerator_buffer(shape=(n, 4, 3, 4, 3), dtype=np.complex128)
S = g.accelerator_buffer(shape=(4, 4, 4), dtype=np.complex128)
C = g.accelerator_buffer(shape=(3, 3, 3), dtype=np.complex128)
G = g.accelerator_buffer(shape=(4, 4), dtype=np.complex128)
S2 = g.accelerator_buffer(shape=(4, 4, 4, 4), dtype=np.complex128)
C2 = g.accelerator_buffer(shape=(3, 3, 3, 3), dtype=np.complex128)

np.random.seed(13)
prop.from_array(np.random.normal(size=prop.shape).astype(prop.dtype))
S.from_array(np.random.normal(size=S.shape).astype(S.dtype))
G.from_array(np.random.normal(size=G.shape).astype(G.dtype))
C.from_array(np.random.normal(size=C.shape).astype(C.dtype))
S2.from_array(np.random.normal(size=S2.shape).astype(S2.dtype))
C2.from_array(np.random.normal(size=C2.shape).astype(C2.dtype))

plan_proton_2pt = g.contract_plan(
    tmp,
    (target, "x"),
    (prop, "x", "s1", "c1", "s2", "c2"),
    (prop, "x", "s3", "c3", "s4", "c4"),
    (prop, "x", "s5", "c5", "s6", "c6"),
    (S, "s1", "s3", "s5"),
    (S, "s2", "s4", "s6"),
    (C, "c1", "c3", "c5"),
    (C, "c2", "c4", "c6"),
)

plan_meson_2pt = g.contract_plan(
    tmp,
    (target, "x"),
    (prop, "x", "s1", "c1", "s2", "c2"),
    (prop, "x", "s3", "c2", "s4", "c1"),
    (G, "s1", "s3"),
    (G, "s2", "s4"),
)

plan_K2pipi = g.contract_plan(
    tmp,
    (target, "x"),
    (prop, "x", "s1", "c1", "s2", "c2"),
    (prop, "x", "s3", "c1", "s4", "c4"),
    (prop, "x", "s5", "c5", "s6", "c6"),
    (prop, "x", "s7", "c5", "s8", "c8"),
    (prop, "x", "s9", "c9", "s10", "c10"),
    (prop, "x", "s11", "c9", "s10", "c10"),
    (G, "s1", "s3"),
    (G, "s5", "s7"),
    (G, "s9", "s11"),
    (S2, "s2", "s4", "s6", "s8"),
    (C2, "c2", "c4", "c6", "c8")    
)

g.message("Temporaries:", tmp)

for tag, plan in [
        ("meson 2pt", plan_meson_2pt),
        ("proton 2pt", plan_proton_2pt),
        ("K2pipi", plan_K2pipi)
]:
    g.message(f"""

   {tag}

""")

    g.message(plan)

    t = g.timer()
    blas = g.blas()
    plan(blas, use_gemm=False)
    blas() # warmup
    
    t("no gemm")
    blas()
    t()

    res1 = target.to_array()
    
    blas = g.blas()
    plan(blas, use_gemm=True)
    blas() # warmup
    
    t("with gemm")
    blas()
    t()

    res2 = target.to_array()
    g.message(t)
    eps = np.linalg.norm(res1 - res2) / np.linalg.norm(res1 + res2)
    g.message(f"Test contraction: {eps}")
    assert eps < 1e-13
