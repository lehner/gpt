#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2021
#
import gpt as g

grid = g.grid([4, 4, 4], g.double)

rng = g.random("test")


# test layers based on 12 densely connected neurons per layer that live on a nearest-neighbor 4^3 grid
# for now real weights only
n_dense = 12
n_depth = 2
n_training = 2

# data type of input layer
# ot_i = g.ot_vector_real_additive_group(n_dense)
ot_i = g.ot_vector_complex_additive_group(n_dense)

# data type of weights
# ot_w = g.ot_matrix_real_additive_group(n_dense)
ot_w = g.ot_matrix_complex_additive_group(n_dense)

n = g.ml.model.sequence([g.ml.layer.nearest_neighbor(grid, ot_i, ot_w)] * n_depth)
W = n.random_weights(rng)

# training_input = [rng.uniform_real(g.lattice(grid, ot_i)) for i in range(n_training)]
# training_output = [rng.uniform_real(g.lattice(grid, ot_i)) for i in range(n_training)]

training_input = [rng.cnormal(g.lattice(grid, ot_i)) for i in range(n_training)]
training_output = [rng.cnormal(g.lattice(grid, ot_i)) for i in range(n_training)]

c = n.cost(training_input, training_output)
g.message("Cost:", c(W))

c.assert_gradient_error(rng, W, [W[0], W[1], W[8], W[9]], 1e-6, 1e-8)

ls0 = g.algorithms.optimize.line_search_none
# ls2 = g.algorithms.optimize.line_search_quadratic
# pr = g.algorithms.optimize.polak_ribiere
# opt = g.algorithms.optimize.non_linear_cg(
#     maxiter=400, eps=1e-7, step=0.3, line_search=ls2, beta=pr
# )

opt = g.algorithms.optimize.gradient_descent(maxiter=40, eps=1e-7, step=1e-7, line_search=ls0)

# Train network
opt(c)(W, W)


# PTC
grid = g.grid([8, 8, 8, 8], g.double)
U = g.qcd.gauge.random(grid, rng)
V = rng.element(g.mcolor(grid))
U_prime = g.qcd.gauge.transformed(U, V)

paths = [
    g.path().forward(0),
    g.path().forward(1),
    g.path().forward(2),
    g.path().forward(3),
    g.path().backward(0),
    g.path().backward(1),
    g.path().backward(2),
    g.path().backward(3),
    g.path().f(0).f(1).b(0).b(1),
]

ot_i = g.ot_vector_spin_color(4, 3)
ot_w = g.ot_matrix_spin(4)

# test equivariance
n = g.ml.model.sequence(
    g.ml.layer.parallel_transport_convolution(grid, U, paths, ot_i, ot_w, 1, 3),
    g.ml.layer.parallel_transport_convolution(grid, U, paths, ot_i, ot_w, 3, 3),
    g.ml.layer.parallel_transport_convolution(grid, U, paths, ot_i, ot_w, 3, 1),
)

n_prime = g.ml.model.sequence(
    g.ml.layer.parallel_transport_convolution(grid, U_prime, paths, ot_i, ot_w, 1, 3),
    g.ml.layer.parallel_transport_convolution(grid, U_prime, paths, ot_i, ot_w, 3, 3),
    g.ml.layer.parallel_transport_convolution(grid, U_prime, paths, ot_i, ot_w, 3, 1),
)

W = n.random_weights(rng)

src = g.vspincolor(grid)
rng.normal(src)
A = n_prime(W, g(V * src))
B = g(V * n(W, src))

eps = (g.norm2(A - B) / g.norm2(A)) ** 0.5
g.message(f"Gauge equivariance test of PTC: {eps} < 1e-13")
assert eps < 1e-13

n_training = 3
training_output = [rng.normal(g.lattice(grid, ot_i)) for i in range(n_training)]
training_input = [rng.normal(g.lattice(grid, ot_i)) for i in range(n_training)]
c = n.cost(training_input, training_output)

c.assert_gradient_error(rng, W, W, 1e-3, 1e-8)

# parallel and sequence layers
n = g.ml.model.sequence(
    g.ml.layer.local_parallel_transport_convolution(grid, U, paths, ot_i, ot_w, 1, 1),
    g.ml.layer.parallel(
        g.ml.layer.sequence(
            g.ml.layer.local_parallel_transport_convolution(grid, U, paths, ot_i, ot_w, 1, 1),
            g.ml.layer.sequence(
                g.ml.layer.local_parallel_transport_convolution(grid, U, paths, ot_i, ot_w, 1, 2),
                g.ml.layer.local_parallel_transport_convolution(grid, U, paths, ot_i, ot_w, 2, 1),
            ),
            g.ml.layer.local_parallel_transport_convolution(grid, U, paths, ot_i, ot_w, 1, 1),
        ),
        g.ml.layer.local_parallel_transport_convolution(grid, U, paths, ot_i, ot_w, 1, 1),
    ),
    g.ml.layer.local_parallel_transport_convolution(grid, U, paths, ot_i, ot_w, 2, 1),
)
W = n.random_weights(rng)
c = n.cost(training_input, training_output)

c.assert_gradient_error(rng, W, W, 1e-3, 1e-8)

# lPTC
n = g.ml.model.sequence(
    g.ml.layer.local_parallel_transport_convolution(grid, U, paths, ot_i, ot_w, 1, 3),
    g.ml.layer.local_parallel_transport_convolution(grid, U, paths, ot_i, ot_w, 3, 3),
    g.ml.layer.local_parallel_transport_convolution(grid, U, paths, ot_i, ot_w, 3, 1),
)
W = n.random_weights(rng)
c = n.cost(training_input, training_output)

c.assert_gradient_error(rng, W, W, 1e-3, 1e-8)

# networks can also generate matrix_operator
matrix = n(W)
test = g(matrix * training_input[0])

# multi-grid networks (I)
coarse_grid = g.grid([4, 4, 4, 4], g.double)
ot_ci = g.ot_vector_spin_color(4, 3)
ot_cw = g.ot_matrix_spin(4)
ot_embedding = g.ot_matrix_spin_color(4, 3)
projector = g.ml.layer.projector_color_trace
get_path_1 = g.ml.layer.parallel_transport_pooling.path.lexicographic
get_path_2 = g.ml.layer.parallel_transport_pooling.path.one_step_lexicographic

Fc = g.qcd.gauge.rectangle(U, 1, 1, trace=False, field=True, real=False)
Fc_prime = g.qcd.gauge.rectangle(U_prime, 1, 1, trace=False, field=True, real=False)

ts = [
    (
        g.ml.layer.parallel_transport_pooling.static_transfer(
            grid, coarse_grid, ot_ci, U, get_path_1
        ),
        g.ml.layer.parallel_transport_pooling.static_transfer(
            grid, coarse_grid, ot_ci, U_prime, get_path_1
        ),
    ),
    (
        g.ml.layer.parallel_transport_pooling.transfer(
            grid,
            coarse_grid,
            ot_ci,
            ot_cw,
            [(U, get_path_1, Fc)],
            ot_embedding=ot_embedding,
            projector=projector,
        ),
        g.ml.layer.parallel_transport_pooling.transfer(
            grid,
            coarse_grid,
            ot_ci,
            ot_cw,
            [(U_prime, get_path_1, Fc_prime)],
            ot_embedding=ot_embedding,
            projector=projector,
        ),
    ),
    (
        g.ml.layer.parallel_transport_pooling.transfer(
            grid,
            coarse_grid,
            ot_ci,
            ot_cw,
            [(U, get_path_1), (U, get_path_2)],
            reference_point=[0, 1, 3, 2],
            ot_embedding=ot_embedding,
            projector=projector,
        ),
        g.ml.layer.parallel_transport_pooling.transfer(
            grid,
            coarse_grid,
            ot_ci,
            ot_cw,
            [(U_prime, get_path_1), (U_prime, get_path_2)],
            reference_point=[0, 1, 3, 2],
            ot_embedding=ot_embedding,
            projector=projector,
        ),
    ),
]

for t, t_prime in ts:
    n = g.ml.model.sequence(
        g.ml.layer.parallel_transport_pooling.project(t),
        g.ml.layer.local_parallel_transport_convolution(
            coarse_grid, t.coarse_gauge, paths, ot_ci, ot_cw, 1, 1
        ),
        g.ml.layer.parallel_transport_pooling.promote(t),
    )
    W = n.random_weights(rng)
    c = n.cost(training_input, training_output)

    g.message("Coarse network weight:", c(W))
    c.assert_gradient_error(rng, W, W, 1e-3, 1e-7)

    n_prime = g.ml.model.sequence(
        g.ml.layer.parallel_transport_pooling.project(t_prime),
        g.ml.layer.local_parallel_transport_convolution(
            coarse_grid, t_prime.coarse_gauge, paths, ot_ci, ot_cw, 1, 1
        ),
        g.ml.layer.parallel_transport_pooling.promote(t_prime),
    )
    A = n_prime(W, g(V * src))
    B = g(V * n(W, src))

    eps = (g.norm2(A - B) / g.norm2(A)) ** 0.5
    g.message(f"Gauge equivariance test of parallel transport pooling model: {eps} < 1e-13")
    assert eps < 1e-13


# multi-grid networks (II)
ncoarse = 4
basis = [rng.normal(g.vspincolor(grid)) for i in range(ncoarse)]
b = g.block.map(coarse_grid, basis)
b.orthonormalize()

ot_ci = g.ot_vector_complex_additive_group(ncoarse)
ot_cw = g.ot_matrix_complex_additive_group(ncoarse)

I = g.complex(coarse_grid)
I[:] = 1
I = [I] * 4

n = g.ml.model.sequence(
    g.ml.layer.block.project(b),
    g.ml.layer.local_parallel_transport_convolution(coarse_grid, I, paths, ot_ci, ot_cw, 1, 1),
    g.ml.layer.block.promote(b),
)
W = n.random_weights(rng)
c = n.cost(training_input, training_output)

g.message("Coarse network weight:", c(W))
c.assert_gradient_error(rng, W, W, 1e-3, 1e-8)
