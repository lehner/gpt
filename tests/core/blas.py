#!/usr/bin/env python3
import gpt as g
import numpy as np

# TODO: test blas for inner_products, block project/promote
# for this need blas routine to work on flat lattice.mview(g.accelerator)


def test_mm_blas(nc, nrhs, precision):
    g.message(f"Test MM {nc}x{nc} with {nrhs} right-hand sides with precision {precision.__name__}")

    # test matrix-matrix multiplication
    grid = g.grid([16, 16, 16, 32], precision)

    A = g.mcomplex(grid, nc)
    B = [g.vcomplex(grid, nc) for _ in range(nrhs)]
    C = [g.vcomplex(grid, nc) for _ in range(nrhs)]
    rng = g.random("test")

    rng.cnormal(A)
    rng.cnormal(B)

    pA = g.pack(A)
    pB = g.pack(B)
    pC = g.pack(C)

    bA = pA.to_accelerator_buffer()
    bB = pB.to_accelerator_buffer()
    bC = pC.to_accelerator_buffer()

    bA = bA.merged_axes(-3, -2)

    # cA = pC.buffer_coordinates(global_coordinates=False)
    idx = pC.buffer_coordinate_indices()

    g.blas().gemm(1.0, bA[idx], bB[idx].T, 0.0, bC[idx].T)()

    pC.from_accelerator_buffer(bC)

    C2 = g.copy(C)
    for x in C2:
        x[:] = 0

    for i in range(nrhs):
        g.eval(C2[i], A * B[i])

    for i in range(nrhs):
        eps = (g.norm2(C[i] - C2[i]) / g.norm2(C[i])) ** 0.5
        g.message(f"Error for rhs[{i}] = {eps}")
        assert eps < precision.eps * 100


test_mm_blas(12, 8, g.double)
test_mm_blas(12, 8, g.single)


# more tests
def run_gemm_versus_numpy(m, k, n, dtype, eps_ref):

    g.message(f"blas versus numpy for {m}x{k}x{n} with dtype={dtype.__name__}")
    A = g.accelerator_buffer(shape=(2, m, k), dtype=dtype)
    A_T = g.accelerator_buffer(shape=(2, k, m), dtype=dtype)
    B = g.accelerator_buffer(shape=(2, k, n), dtype=dtype)
    B_T = g.accelerator_buffer(shape=(2, n, k), dtype=dtype)
    C = g.accelerator_buffer(shape=(2, m, n), dtype=dtype)
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
    idx = np.arange(2, dtype=np.int64)

    def check(tag):
        eps = np.linalg.norm(_AB - C.to_array()) / np.linalg.norm(_AB)
        g.message(f"Check {tag}: {eps}")
        assert eps < eps_ref

    _AB = np.einsum("xij,xjk->xik", _A, _B)
    g.blas().gemm(1.0, A[idx], B[idx], 0.0, C[idx])()
    check("A B = C")

    _AB = np.einsum("xji,xjk->xik", _A_T, _B)
    g.blas().gemm(1.0, A_T[idx].T, B[idx], 0.0, C[idx])()
    check("A.T B = C")

    _AB = np.einsum("xji,xkj->xik", _A_T, _B_T)
    g.blas().gemm(1.0, A_T[idx].T, B_T[idx].T, 0.0, C[idx])()
    check("A.T B.T = C")

    _AB = np.einsum("xij,xkj->xik", _A, _B_T)
    g.blas().gemm(1.0, A[idx], B_T[idx].T, 0.0, C[idx])()
    check("A B.T = C")

    _AB = np.einsum("xji,xjk->xik", np.conjugate(_A_T), _B)
    g.blas().gemm(1.0, A_T[idx].H, B[idx], 0.0, C[idx])()
    check("A.H B = C")

    _AB = np.einsum("xij,xkj->xik", _A, np.conjugate(_B_T))
    g.blas().gemm(1.0, A[idx], B_T[idx].H, 0.0, C[idx])()
    check("A B.H = C")

    _AB = np.einsum("xji,xkj->xik", np.conjugate(_A_T), np.conjugate(_B_T))
    g.blas().gemm(1.0, A_T[idx].H, B_T[idx].H, 0.0, C[idx])()
    check("A.H B.H = C")


run_gemm_versus_numpy(4, 6, 8, np.complex128, 1e-14)
run_gemm_versus_numpy(4, 6, 8, np.complex64, 1e-6)
