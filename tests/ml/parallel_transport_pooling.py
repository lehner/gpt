#!/usr/bin/env python3
import gpt as g
import numpy as np

grid = g.grid([8, 8, 8, 16], g.double)
coarse_grid = g.grid([4, 4, 4, 4], g.double)
ot_ci = g.ot_vector_spin_color(4, 3)
ot_cw = g.ot_matrix_spin(4)

rng = g.random("test")

U = g.qcd.gauge.random(grid, rng)

training_input = rng.cnormal([g.vspincolor(coarse_grid)])
training_output = rng.cnormal([g.vspincolor(grid)])

t = g.ml.layer.parallel_transport_pooling.transfer(
    grid,
    coarse_grid,
    ot_ci,
    ot_cw,
    [
        (U, g.ml.layer.parallel_transport_pooling.path.lexicographic),
        (U, g.ml.layer.parallel_transport_pooling.path.one_step_lexicographic),
    ],
    ot_embedding=g.ot_matrix_spin_color(4, 3),
    projector=g.ml.layer.projector_color_trace,
)

n = g.ml.model.sequence(g.ml.layer.parallel_transport_pooling.promote(t))

W = n.random_weights(rng)
c = n.cost(training_input, training_output)

g.message("Coarse network weight:", c(W))
c.assert_gradient_error(rng, W, W, 1e-3, 1e-7)


n = g.ml.model.sequence(g.ml.layer.parallel_transport_pooling.project(t))

W = n.random_weights(rng)
c = n.cost(training_output, training_input)

g.message("Coarse network weight:", c(W))
c.assert_gradient_error(rng, W, W, 1e-3, 1e-7)


n = g.ml.model.sequence(
    g.ml.layer.parallel_transport_pooling.project(t),
    g.ml.layer.parallel_transport_pooling.promote(t.cloned()),
)

W = n.random_weights(rng)
c = n.cost(rng.cnormal(g.copy(training_output)), training_output)

g.message("Coarse network weight:", c(W))
c.assert_gradient_error(rng, W, W, 1e-3, 1e-6)


n = g.ml.model.sequence(
    g.ml.layer.parallel_transport_pooling.project(t),
    g.ml.layer.parallel_transport_pooling.promote(t),
)

W = n.random_weights(rng)
c = n.cost(rng.cnormal(g.copy(training_output)), training_output)

g.message("Coarse network weight:", c(W))
c.assert_gradient_error(rng, W, W, 1e-3, 1e-5)
