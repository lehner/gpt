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
ot_i = g.ot_vector_real_additive_group(n_dense)

# data type of weights
ot_w = g.ot_matrix_real_additive_group(n_dense)

n = g.ml.network.feed_forward([g.ml.layer.nearest_neighbor(grid, ot_i, ot_w)] * n_depth)
W = n.random_weights(rng)

training_input = [rng.uniform_real(g.lattice(grid, ot_i)) for i in range(n_training)]
training_output = [rng.uniform_real(g.lattice(grid, ot_i)) for i in range(n_training)]

c = n.cost(training_input, training_output)
g.message("Cost:", c(W))

c.assert_gradient_error(rng, W, [W[0], W[1]], 1e-4, 1e-8)

ls0 = g.algorithms.optimize.line_search_none
# ls2 = g.algorithms.optimize.line_search_quadratic
# pr = g.algorithms.optimize.polak_ribiere
# opt = g.algorithms.optimize.non_linear_cg(
#     maxiter=40, eps=1e-7, step=1e-1, line_search=ls2, beta=pr
# )

opt = g.algorithms.optimize.gradient_descent(
    maxiter=40, eps=1e-7, step=0.5, line_search=ls0
)

# Train network
opt(c)(W, W)
