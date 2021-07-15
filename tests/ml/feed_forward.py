#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g

grid = g.grid([4, 4, 4], g.double)

rng = g.random("test")
i0 = rng.normal(g.complex(grid))

n = g.ml.network.feed_forward([g.ml.layer.nearest_neighbor(grid)] * 2)
W = n.random_weights(rng)

g.message("Feed forward", n(W, i0)[0, 0, 0])

training_input = [rng.uniform_real(g.complex(grid)) for i in range(2)]
training_output = [rng.uniform_real(g.complex(grid)) for i in range(2)]

c = n.cost(training_input, training_output)
g.message("Cost:", c(W))

c.assert_gradient_error(rng, W, W, 1e-4, 1e-8)

ls0 = g.algorithms.optimize.line_search_none
ls2 = g.algorithms.optimize.line_search_quadratic
pr = g.algorithms.optimize.polak_ribiere
opt = g.algorithms.optimize.non_linear_cg(
    maxiter=40, eps=1e-7, step=1e-1, line_search=ls2, beta=pr
)

# opt = g.algorithms.optimize.gradient_descent(
#    maxiter=4000, eps=1e-7, step=0.2, line_search=ls0
# )

# Train network
opt(c)(W, W)
