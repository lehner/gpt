#!/usr/bin/env python3
import gpt as g
import numpy as np

# TODO: test blas for inner_products, block project/promote
# for this need blas routine to work on flat lattice.mview(g.accelerator)

# rank_fft test
np.random.seed(13)
bufa = np.random.normal(size=(4, 3, 48)).astype(np.complex128)
buf = g.accelerator.buffer(bufa)
tar = g.accelerator.buffer(bufa)
k = g.accelerator.kernel().rank_fft(buf, tar, True)()
tara = np.fft.fft(bufa)

eps = np.linalg.norm(tar.to_array() - tara) / np.linalg.norm(tara)
g.message(f"FFT test: {eps}")
assert eps < 1e-13

k = g.accelerator.kernel().rank_fft(buf, tar, False)()
tara = np.fft.ifft(bufa, norm="forward")

eps = np.linalg.norm(tar.to_array() - tara) / np.linalg.norm(tara)
g.message(f"iFFT test: {eps}")
assert eps < 1e-13


def test_mm_blas(nc, nrhs, precision):
    g.message(f"Test MM {nc}x{nc} with {nrhs} right-hand sides with precision {precision.__name__}")

    # test matrix-matrix multiplication
    L = [16, 16, 16, 32]
    grid = g.grid(L, precision)
    rL = list(reversed(grid.ldimensions))

    A = g.mcomplex(grid, nc)
    B = [g.vcomplex(grid, nc) for _ in range(nrhs)]
    C = [g.vcomplex(grid, nc) for _ in range(nrhs)]
    rng = g.random("test")

    rng.cnormal(A)
    rng.cnormal(B)
    rng.cnormal(C)  # just initialize some values (need to avoid nan)

    pA = g.pack(A)
    pB = g.pack(B)
    pC = g.pack(C)

    margin = [1, 1, 1, 1]
    bA = pA.to_accelerator_buffer()
    bB = pB.to_accelerator_buffer(margin=margin)
    bC = pC.to_accelerator_buffer()

    bA = bA.merged_axes(-3, -2).split_axis(-1, 4, 3).merged_axes(-2, -1)

    # test transposition on device versus on host
    eps = np.linalg.norm(
        bA.transpose(4, 5, 3, 2, 1, 0).to_array() - bA.to_array().transpose(4, 5, 3, 2, 1, 0)
    )
    assert eps == 0.0

    # self-consistent transposition with blas
    tr = bA.transpose(4, 5, 3, 2, 1, 0)
    tr2 = g.accelerator.buffer(tr)
    g.accelerator.kernel().transpose(tr2, bA, (4, 5, 3, 2, 1, 0))()
    eps = np.linalg.norm(tr.to_array() - tr2.to_array())
    assert eps == 0.0

    cA = bA.coordinates(range(4))
    cB = bB.coordinates(range(4))

    bulkB = bB.bulk(cB, margin=margin)

    halo_exchange = bB.halo_exchange(grid, margin=margin, max_point_sqr=1)

    halo_exchange()

    idxA = bA.indices(range(4))
    idxB = bB.indices(range(4), shift=[0, 0, 1, 0])[bulkB]

    # test cshift of indices
    for sh in [[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [1, -1, 3, 2]]:
        idxA2 = bA.indices(range(4), shift=sh)
        eps = np.linalg.norm(np.mod(cA[idxA2] - cA[idxA] + rL - sh, rL))
        g.message(f"Test cshift with {sh}: {eps}")
        assert eps < 1e-14

    g.accelerator.kernel().gemm(1.0, bA[idxA], bB[idxB].T, 0.0, bC[idxA].T)()

    pC.from_accelerator_buffer(bC)

    C2 = g.copy(C)
    for x in C2:
        x[:] = 0

    for i in range(nrhs):
        g.eval(C2[i], A * g.cshift(B[i], 1, 1))

    for i in range(nrhs):
        eps = (g.norm2(C[i] - C2[i]) / g.norm2(C[i])) ** 0.5
        g.message(f"Error for rhs[{i}] = {eps}")
        if eps > precision.eps * 100:
            for y in range(L[1]):
                g.message(y, C[i][0, y, 0, 0, 0], C2[i][0, y, 0, 0, 0])
            assert False


test_mm_blas(12, 8, g.single)
test_mm_blas(12, 8, g.double)


# more tests
def run_gemm_versus_numpy(m, k, n, dtype, eps_ref):

    g.message(f"blas versus numpy for {m}x{k}x{n} with dtype={dtype.__name__}")
    A = g.accelerator.buffer(shape=(2, m, k), dtype=dtype)
    A_T = g.accelerator.buffer(shape=(2, k, m), dtype=dtype)
    B = g.accelerator.buffer(shape=(2, k, n), dtype=dtype)
    B_T = g.accelerator.buffer(shape=(2, n, k), dtype=dtype)
    C = g.accelerator.buffer(shape=(2, m, n), dtype=dtype)
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
    g.accelerator.kernel().gemm(1.0, A[idx], B[idx], 0.0, C[idx])()
    check("A B = C")

    _AB = np.einsum("xji,xjk->xik", _A_T, _B)
    g.accelerator.kernel().gemm(1.0, A_T[idx].T, B[idx], 0.0, C[idx])()
    check("A.T B = C")

    _AB = np.einsum("xji,xkj->xik", _A_T, _B_T)
    g.accelerator.kernel().gemm(1.0, A_T[idx].T, B_T[idx].T, 0.0, C[idx])()
    check("A.T B.T = C")

    _AB = np.einsum("xij,xkj->xik", _A, _B_T)
    g.accelerator.kernel().gemm(1.0, A[idx], B_T[idx].T, 0.0, C[idx])()
    check("A B.T = C")

    _AB = np.einsum("xji,xjk->xik", np.conjugate(_A_T), _B)
    g.accelerator.kernel().gemm(1.0, A_T[idx].H, B[idx], 0.0, C[idx])()
    check("A.H B = C")

    _AB = np.einsum("xij,xkj->xik", _A, np.conjugate(_B_T))
    g.accelerator.kernel().gemm(1.0, A[idx], B_T[idx].H, 0.0, C[idx])()
    check("A B.H = C")

    _AB = np.einsum("xji,xkj->xik", np.conjugate(_A_T), np.conjugate(_B_T))
    g.accelerator.kernel().gemm(1.0, A_T[idx].H, B_T[idx].H, 0.0, C[idx])()
    check("A.H B.H = C")


run_gemm_versus_numpy(4, 6, 8, np.complex128, 1e-14)
run_gemm_versus_numpy(4, 6, 8, np.complex64, 1e-6)

# test tensor <> accelerator_buffer
for i in [0, 1, 2, 3, 5]:
    eps = np.linalg.norm(
        g.accelerator.buffer(g.gamma[i].tensor()).to_array() - g.gamma[i].tensor().array
    )
    g.message(f"Tensor <> accelerator_buffer test: {eps}")
    assert eps < 1e-13

# test matrix inverse
L = [16, 16, 16, 32]
rng = g.random("test")
for precision in [g.single, g.double]:
    grid = g.grid(L, precision)

    local_buffer_shape = list(reversed(grid.ldimensions))

    # test indexed sum
    M = g.mcolor(grid)
    rng.cnormal(M)
    A = g.pack(M).to_accelerator_buffer()
    indices_numpy = (
        A.coordinates(range(4))[:, 1].reshape(local_buffer_shape)
        + grid.processor_coor[2] * grid.ldimensions[2]
    )
    indices = g.accelerator.buffer(indices_numpy)
    target = g.accelerator.buffer(np.zeros(shape=(grid.gdimensions[2],), dtype=A.dtype))
    tr = g.accelerator.buffer(shape=local_buffer_shape, dtype=A.dtype)
    blas = g.accelerator.kernel()
    blas.indexed_sum(A, indices, target)
    blas.indexed_sum(A, indices, target, accumulate=True)
    blas()

    ab_is = grid.globalsum(target.to_array())
    ref_is = [2 * np.sum(x.array) for x in g.slice(M, 2)]
    eps = np.linalg.norm(ab_is - np.array(ref_is)) / np.linalg.norm(ab_is)
    g.message(f"Test indexed sum: {eps}")
    assert eps < precision.eps * 50

    # test contraction with complex conjugation
    blas = g.accelerator.kernel()
    blas.contract(
        (tr, "x", "y", "z", "t"),
        (A, "x", "y", "z", "t", "n", "c1", "c2"),
        (A, "x", "y", "z", "t", "n", "c1", "c2", "*"),
    )
    blas()

    es = g.einsum("aa->", M, g.complex(grid))
    x = es(M)

    es = g.einsum(
        "xyztncc->xyzt",
        g.accelerator.buffer(M),
        g.accelerator.buffer(shape=tr.shape, dtype=grid.precision.complex_dtype),
    )
    y = es(g.accelerator.buffer(M))

    eps = (
        g.norm2(g.trace(M * g.adj(M)) - g.pack(g.complex(grid)).from_accelerator_buffer(tr)) ** 0.5
        / g.norm2(g.trace(M * g.adj(M))) ** 0.5
    )
    g.message(f"Test expression <> contract: {eps}")
    assert eps < precision.eps * 50

    eps = g.norm2(x - g.pack(g.complex(grid)).from_accelerator_buffer(y))
    g.message(f"Test einsum(accelerator_buffer) <> einsum(lattice): {eps}")
    assert eps < precision.eps * 50

    # test indexed sum
    sref = g.slice(g.trace(M * g.adj(M)), 3)
    blas = g.accelerator.kernel()
    corr = g.accelerator.buffer(shape=(grid.gdimensions[3],), dtype=grid.precision.complex_dtype)
    pln = g.contract.plan(
        g.accelerator.buffer_manager(),
        (corr, "t_global"),
        (
            g.contract.indexed_sum(index=grid.local_indices(3), length=corr.shape[0]),
            "t_global",
            "t",
        ),
        (g.accelerator.buffer(M), "t", "z", "y", "x", "n", "c1", "c2"),
        (g.accelerator.buffer(M), "t", "z", "y", "x", "n", "c1", "c2", "*"),
    )
    g.message(pln)
    pln(blas)
    # string representation of kernels queued up in blas
    g.message(f"Blas plan:\n{blas}")
    # execut kernels
    blas()
    scon = grid.globalsum(corr.to_array())
    for x, y in zip(sref, scon):
        eps = abs(x - y) / abs(x)
        g.message(f"Test indexed sum in contract.plan: {eps}")
        assert eps < grid.precision.eps * 50

    # second indexed sum test
    sref = g.slice(M, 3)
    blas = g.accelerator.kernel()
    corr = g.accelerator.buffer(
        shape=(grid.gdimensions[3], 3, 3), dtype=grid.precision.complex_dtype
    )
    pln = g.contract.plan(
        g.accelerator.buffer_manager(),
        (corr, "t_global", "c1", "c2"),
        (
            g.contract.indexed_sum(index=grid.local_indices(3), length=corr.shape[0]),
            "t_global",
            "t",
        ),
        (g.accelerator.buffer(M), "t", "z", "y", "x", "n", "c1", "c2"),
    )
    g.message(pln)
    pln(blas)
    # string representation of kernels queued up in blas
    g.message(f"Blas plan:\n{blas}")
    # execut kernels
    blas()
    scon = grid.globalsum(corr.to_array())
    for x, y in zip(sref, scon):
        eps = np.linalg.norm(x.array - y) / np.linalg.norm(y)
        g.message(f"Test indexed sum (2) in contract.plan: {eps}")
        assert eps < grid.precision.eps * 50

    # test complex matrix
    M = g.mcomplex(grid, 16)
    rng.cnormal(M)

    # test determinant
    A = g.pack(M).to_accelerator_buffer()
    idx = A.indices([0, 1, 2, 3])

    t = g.timer("determinant")
    t("blas")
    C = A.empty_clone(A.shape[0:-2])
    g.accelerator.kernel().det(A[idx], C[idx])()
    Mdet = g.complex(grid)
    g.pack(Mdet).from_accelerator_buffer(C)

    t("numpy")
    cache = {}
    Mdet_ref = g.complex(grid)
    Mdet_ref[:] = np.linalg.det(M[:, cache])

    t("numpy cache")
    Mdet_ref = g.complex(grid)
    Mdet_ref[:] = np.linalg.det(M[:, cache])
    t()

    eps2 = g.norm2(Mdet - Mdet_ref) / g.norm2(Mdet)
    g.message("DET", eps2)
    assert eps2 < precision.eps**2 * 100

    g.message(t)

    # test inverse
    A = g.pack(M).to_accelerator_buffer()
    idx = A.indices([0, 1, 2, 3])

    t = g.timer("inverse")
    t("blas_setup")
    C = A.empty_clone()
    blas = g.accelerator.kernel().inv(
        A[idx], C[idx]
    )  # inv and det do not guarantee that A remains unchanged
    t("blas_exec")
    blas()
    t("blas_setup")
    Minv = g.lattice(M)
    g.pack(Minv).from_accelerator_buffer(C)
    t("grid")
    g(g.matrix.inv(M))
    t()

    g.message(t)
    eps2 = g.norm2(M * Minv - g.identity(M)) / g.norm2(M)
    g.message("INVERSE", eps2)
    assert eps2 < precision.eps**2 * 100


# test general tensor contractions

tmp = g.accelerator.buffer_manager()
target = g.accelerator.buffer(shape=(18,), dtype=np.complex128)
prop = g.accelerator.buffer(shape=(18, 4, 3, 4, 3), dtype=np.complex128)
S = g.accelerator.buffer(shape=(4, 4, 4), dtype=np.complex128)
C = g.accelerator.buffer(shape=(3, 3, 3), dtype=np.complex128)

np.random.seed(13)
prop.from_array(np.random.normal(size=prop.shape).astype(prop.dtype))
S.from_array(np.random.normal(size=S.shape).astype(S.dtype))
C.from_array(np.random.normal(size=C.shape).astype(C.dtype))

plan = g.contract.plan(
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

# do it again to test re-use in tmp
plan = g.contract.plan(
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

g.message(plan)
g.message(tmp)

t = g.timer()
t("slow")
blas = g.accelerator.kernel()
plan(blas, optimal=False)
blas()
t()

res1 = target.to_array()

t("fast")
blas = g.accelerator.kernel()
plan(blas)
blas()
t()

res2 = target.to_array()
g.message(t)
eps = np.linalg.norm(res1 - res2) / np.linalg.norm(res1 + res2)
g.message(f"Test proton 2pt contraction: {eps}")
assert eps < 1e-13
