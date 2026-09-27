#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2026
#
# Expression evaluation where the target is also an operand (in-place
# evaluation), and linear combinations of terms carrying different unary
# operators, checked
#   1) in place against out of place (bitwise), and
#   2) against an independent numpy reference.
#
# Covers otypes stored in a single and in several v_obj (large singlet
# matrices / vectors), where the evaluation is split into several passes.
#
import gpt as g
import numpy as np
import sys

grid_shape = [4, 4, 4, 4]
rng = g.random("expression_aliasing")


def matrix_types():
    return {
        "mcolor": g.ot_matrix_color(3),
        "mspincolor": g.ot_matrix_spin_color(4, 3),
        "msinglet4": g.ot_matrix_singlet(4),
        "msinglet8": g.ot_matrix_singlet(8),
        "msinglet12": g.ot_matrix_singlet(12),
        "adjoint_su3": g.ot_matrix_su_n_adjoint_algebra(3),
    }


def vector_types():
    return {
        "complex": g.ot_singlet(),
        "vcolor": g.ot_vector_color(3),
        "vspincolor": g.ot_vector_spin_color(4, 3),
        "vsinglet12": g.ot_vector_singlet(12),
    }


# (label, expression of (z, y), accumulate?)
common_forms = [
    ("z @= conj(z)", lambda z, y: g.conj(z), False),
    ("z @= z + conj(z)", lambda z, y: z + g.conj(z), False),
    ("z @= 2z - 1j conj(z)", lambda z, y: 2.0 * z - 1j * g.conj(z), False),
    ("z += conj(z)", lambda z, y: g.conj(z), True),
    ("z += z + conj(z)", lambda z, y: z + g.conj(z), True),
    ("z @= 0.5z + (0.2-1j) y", lambda z, y: 0.5 * z + (0.2 - 1j) * y, False),
]

matrix_forms = common_forms + [
    ("z @= transpose(z)", lambda z, y: g.transpose(z), False),
    ("z @= adj(z)", lambda z, y: g.adj(z), False),
    ("z @= z + adj(z)", lambda z, y: z + g.adj(z), False),
    ("z @= 2z - 1j transpose(z)", lambda z, y: 2.0 * z - 1j * g.transpose(z), False),
    ("z @= transpose(z) + conj(z)", lambda z, y: g.transpose(z) + g.conj(z), False),
    (
        "z @= 0.3j z + (0.2-1j) adj(y) + conj(y) - 0.7 transpose(z)",
        lambda z, y: 0.3j * z + (0.2 - 1j) * g.adj(y) + g.conj(y) - 0.7 * g.transpose(z),
        False,
    ),
    ("z += adj(z)", lambda z, y: g.adj(z), True),
    ("z += 0.5 transpose(z) - y", lambda z, y: 0.5 * g.transpose(z) - y, True),
    ("z @= z * z", lambda z, y: z * z, False),
    ("z @= adj(z) * z", lambda z, y: g.adj(z) * z, False),
    ("z @= z * adj(z) + z", lambda z, y: z * g.adj(z) + z, False),
]

vector_forms = common_forms


def lattice_of(grid, otype):
    x = g.lattice(grid, otype)
    rng.cnormal(x)
    return x


def bitwise_equal(a, b):
    return np.array_equal(a[:].view(np.uint8), b[:].view(np.uint8))


# 1) in place versus out of place
for precision in [g.double, g.single]:
    grid = g.grid(grid_shape, precision)
    for types, forms in [(matrix_types(), matrix_forms), (vector_types(), vector_forms)]:
        for tname, otype in types.items():
            for label, f, accumulate in forms:
                z = lattice_of(grid, otype)
                y = lattice_of(grid, otype)
                z_in = g.copy(z)
                if accumulate:
                    # same association as the in-place z += f: accumulate into
                    # an independent target
                    expected = g.copy(z)
                    expected += f(z_in, y)
                    z += f(z, y)
                else:
                    expected = g(f(z_in, y))
                    g.eval(z, f(z, y))
                ok = bitwise_equal(z, expected)
                g.message(
                    f"In place vs out of place, {precision.__name__} {tname} ({len(z.v_obj)} v_obj): {label}: {'ok' if ok else 'MISMATCH'}"
                )
                assert ok


# other kernel families that read across elements of a site
grid = g.grid(grid_shape, g.double)
psi = lattice_of(grid, g.ot_vector_spin_color(4, 3))
for mu in [0, 1, 2, 3, 5]:
    expected = g(g.gamma[mu] * psi + psi)
    psi_in_place = g.copy(psi)
    g.eval(psi_in_place, g.gamma[mu] * psi_in_place + psi_in_place)
    assert bitwise_equal(psi_in_place, expected)
T = g.mcolor([[complex(i + 2 * j, i - j) for j in range(3)] for i in range(3)])
for f in [lambda z: T * z, lambda z: z * T]:
    m = lattice_of(grid, g.ot_matrix_color(3))
    expected = g(f(g.copy(m)))
    g.eval(m, f(m))
    assert bitwise_equal(m, expected)
g.message("In place gamma and tensor multiplication: ok")


# 2) linear combinations with unary terms against numpy
def np_transpose(a):
    # site index first, then pairs of (row, column) indices per tensor level
    n = (a.ndim - 1) // 2
    axes = [0] + [x for k in range(n) for x in (2 * k + 2, 2 * k + 1)]
    return np.transpose(a, axes)


np_unary = {
    "id": lambda a: a,
    "conj": np.conj,
    "transpose": np_transpose,
    "adj": lambda a: np.conj(np_transpose(a)),
}
g_unary = {"id": lambda a: a, "conj": g.conj, "transpose": g.transpose, "adj": g.adj}

combinations = [
    [(1.0, "adj", "x")],
    [(1.0, "conj", "x")],
    [(1.0, "transpose", "x")],
    [(0.5, "id", "x"), (-0.5, "adj", "x")],
    [(0.3j, "id", "x"), (0.2 - 1j, "adj", "y"), (1.0, "conj", "y"), (-0.7, "transpose", "x")],
    [(2.0, "transpose", "y"), (-1.5j, "conj", "x"), (0.25, "id", "y")],
]

for precision, tol in [(g.double, 1e-14), (g.single, 1e-6)]:
    grid = g.grid(grid_shape, precision)
    for tname, otype in matrix_types().items():
        x = lattice_of(grid, otype)
        y = lattice_of(grid, otype)
        operands = {"x": x, "y": y}
        np_operands = {"x": x[:], "y": y[:]}
        for comb in combinations:
            expr = None
            ref = 0
            for coef, unary, name in comb:
                term = coef * g_unary[unary](operands[name])
                expr = term if expr is None else expr + term
                ref = ref + coef * np_unary[unary](np_operands[name])
            result = g(expr)[:]
            eps = np.max(np.abs(result - ref)) / np.max(np.abs(ref))
            label = " + ".join(f"{c}*{u}({n})" for c, u, n in comb)
            g.message(f"Numpy reference, {precision.__name__} {tname}: {label}: {eps}")
            assert eps < tol

g.message("All tests passed")
