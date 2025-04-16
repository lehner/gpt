#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2025
#
import gpt as g
import numpy as np

g.default.set_verbose("random", False)
rng = g.random("benchmark", "vectorized_ranlux24_24_64")
L = g.default.get_ivec("--grid", [16,16,16,32], 4)
N = g.default.get_int("--N", 100)
m = g.default.get_int("--m", 40)
k = g.default.get_int("--k", 40)
n = g.default.get_int("--n", 12)

for precision in [g.single, g.double]:
    grid = g.grid(L, precision)
    
    g.message(
        f"""


    blas.gemm benchmark with
    fdimensions  : {grid.fdimensions}
    m            : {m}
    k            : {k}
    n            : {n}
    precision    : {precision.__name__}
"""
    )
    
    # test matrix-matrix multiplication
    flops_per_matrix_vector_multiply = m * (k * 6 + (k - 1) * 2) 
    flops = flops_per_matrix_vector_multiply * grid.gsites * N * n
    nbytes = (m * k + k * n + m * n) * precision.nbytes * 2 * N * grid.gsites

    dtype = precision.complex_dtype
    sites = int(grid.gsites // grid.Nprocessors)
    A = g.accelerator_buffer(shape=(sites, m, k), dtype=dtype)
    A_T = g.accelerator_buffer(shape=(sites, k, m), dtype=dtype)
    B = g.accelerator_buffer(shape=(sites, k, n), dtype=dtype)
    B_T = g.accelerator_buffer(shape=(sites, n, k), dtype=dtype)
    C = g.accelerator_buffer(shape=(sites, m, n), dtype=dtype)
    _A = A.to_array()
    _B = B.to_array()
    _C = C.to_array()
    _A_T = A_T.to_array()
    _B_T = B_T.to_array()
    np.random.seed(13)
    _A[:] = np.random.randn(*A.shape) + 1j * np.random.randn(*A.shape)
    _B[:] = np.random.randn(*B.shape) + 1j * np.random.randn(*B.shape)
    _C[:] = np.random.randn(*C.shape) + 1j * np.random.randn(*C.shape)
    _A_T[:] = np.random.randn(*A_T.shape) + 1j * np.random.randn(*A_T.shape)
    _B_T[:] = np.random.randn(*B_T.shape) + 1j * np.random.randn(*B_T.shape)
    A.from_array(_A)
    B.from_array(_B)
    C.from_array(_C)
    A_T.from_array(_A_T)
    B_T.from_array(_B_T)
    idx = np.arange(sites, dtype=np.int64)

    j = g.blas().gemm(1.0, A[idx], B[idx], 0.0, C[idx])

    # warmup
    for _ in range(5):
        j()

    t0 = g.time()
    for _ in range(N):
        j()
    t1 = g.time()

    # Report
    GFlopsPerSec = flops / (t1 - t0) / 1e9
    GBPerSec = nbytes / (t1 - t0) / 1e9
    g.message(
        f"""
{N} applications of A_mk B_kn = C_mn
    Time to complete            : {t1-t0:.2f} s
    Total performance           : {GFlopsPerSec:.2f} GFlops/s
    Total bandwidth             : {GBPerSec:.2f} GB/s
    """)

    j = g.blas().gemm(1.0, A_T[idx].H, B[idx], 0.0, C[idx])

    # warmup
    for _ in range(5):
        j()

    t0 = g.time()
    for _ in range(N):
        j()
    t1 = g.time()

    # Report
    GFlopsPerSec = flops / (t1 - t0) / 1e9
    GBPerSec = nbytes / (t1 - t0) / 1e9
    g.message(
        f"""
{N} applications of adj(A)_mk B_kn = C_mn 
    Time to complete            : {t1-t0:.2f} s
    Total performance           : {GFlopsPerSec:.2f} GFlops/s
    Total bandwidth             : {GBPerSec:.2f} GB/s
    """)

    j = g.blas().gemm(1.0, A[idx], B_T[idx].H, 0.0, C[idx])

    # warmup
    for _ in range(5):
        j()

    t0 = g.time()
    for _ in range(N):
        j()
    t1 = g.time()

    # Report
    GFlopsPerSec = flops / (t1 - t0) / 1e9
    GBPerSec = nbytes / (t1 - t0) / 1e9
    g.message(
        f"""
{N} applications of A_mk adj(B)_kn = C_mn 
    Time to complete            : {t1-t0:.2f} s
    Total performance           : {GFlopsPerSec:.2f} GFlops/s
    Total bandwidth             : {GBPerSec:.2f} GB/s
    """)
