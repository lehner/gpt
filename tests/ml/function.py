#!/usr/bin/env python3
#
# The base class g.ml.function: named slots, owned parameters, plain and node
# evaluation, type checks, gradients and training through the optimizers.
#
import gpt as g
import numpy as np

rad = g.ad.reverse
rng = g.random("test")
grid = g.grid([4, 4, 4, 4], g.double)


class example(g.ml.function):
    # y = Re(a) x + b + s c_0 c_1 x + d_0 x x + d_1 x
    # with a real scalar a, a complex field b, a list slot c of two complex
    # fields, a numpy array d, and the constant s
    def __init__(self, grid):
        x = g.complex(grid)
        super().__init__(
            inputs=[("x", x)],
            outputs=[("y", x)],
            parameters=[
                ("a", 0j),
                ("b", g.complex(grid)),
                ("c", [g.complex(grid), g.complex(grid)]),
                ("d", np.zeros(2, dtype=np.complex128)),
            ],
            constants=[("s", 0.5)],
        )

    def initialize(self, rng):
        self["a"] = 0.8 + 0j
        for name in ["b", "c.0", "c.1"]:
            rng.cnormal(self[name], sigma=0.3)
        self["d"] = np.array([0.1 + 0.2j, -0.3 + 0.1j])

    def evaluate(self, inputs, parameters, constants):
        (x,) = inputs
        a, b, c, d = parameters
        (s,) = constants
        y = g.component.real(a) * x + b + c[0] * c[1] * x * s + d[0] * x * x + d[1] * x
        return [g(y)]


def reference(f, x):
    a, b, c0, c1, d = f.parameters()
    s = f["s"]
    return g(a.real * x + b + s * c0 * c1 * x + d[0] * x * x + d[1] * x)


f = example(grid)
f.initialize(rng)

# names and storage
assert f.input_names() == ["x"] and f.output_names() == ["y"]
assert f.parameter_names() == ["a", "b", "c.0", "c.1", "d"]
assert f.constant_names() == ["s"]
p = f.parameters()
assert f["c"][0] is p[2] and f["c"][1] is p[3] and f["c.1"] is p[3] and f["d"] is p[4]
assert f["s"] == 0.5

# assignment: in place for fields, replacement for numbers and arrays
b = f["b"]
new_b = rng.cnormal(g.complex(grid))
f["b"] = new_b
assert f["b"] is b and g.norm2(b - new_b) == 0.0
f["a"] = 0.7 + 0j
assert f.parameters()[0] == 0.7
f["c"] = [g.copy(f["c.1"]), g.copy(f["c.0"])]
assert f["c"][0] is p[2]
g.message("Names and storage: ok")

# invalid declarations
for parameters, constants in [
    ([("a.b", 0j)], []),
    ([("3", 0j)], []),
    ([("a", 0j), ("a", 0j)], []),
    ([("a", 0j)], [("a", 1.0)]),
    ([("", 0j)], []),
]:
    try:
        g.ml.function([], [], parameters, constants)
        assert False
    except ValueError as e:
        g.message(f"Rejected: {e}")

# plain evaluation
x = rng.cnormal(g.complex(grid))
y = f([x])[0]
eps2 = g.norm2(y - reference(f, x)) / g.norm2(y)
g.message(f"Plain evaluation: {eps2}")
assert eps2 < 1e-28

# node evaluation (parameters and input as nodes) gives the same value
nx = rad.node(x)
leaves = [rad.node(v) for v in f.parameters()]
ny = f([nx], leaves)[0]
eps2 = g.norm2(ny(with_gradients=False) - y) / g.norm2(y)
g.message(f"Node evaluation: {eps2}")
assert eps2 < 1e-28

# types are checked for nodes, not for plain values
xr = g.real(grid)
xr[:] = 1
for call, expected in [
    (lambda: f([rad.node(xr)]), TypeError),
    (lambda: f([nx], [rad.node(g.real(grid))] + leaves[1:]), TypeError),
    (lambda: f([nx], leaves[:-1]), ValueError),
    (lambda: f([nx, nx]), ValueError),
]:
    try:
        call()
        assert False
    except expected as e:
        g.message(f"Rejected: {e}")
f([xr])

# gradients w.r.t. all parameters: the leaf gradient is dL/dRe + i dL/dIm, so
# the derivative along a direction v is Re <gradient, v>
t = rng.cnormal(g.complex(grid))


def loss(parameters, x=x):
    return g.norm2(f([x], parameters)[0] - t)


leaves = [rad.node(v) for v in f.parameters()]
loss(leaves)()
gradients = [n.gradient for n in leaves]
assert gradients[0].imag == 0.0
directions = [
    0.3 - 0.2j,
    rng.cnormal(g.complex(grid)),
    rng.cnormal(g.complex(grid)),
    rng.cnormal(g.complex(grid)),
    np.array([0.4 + 0.1j, -0.2 + 0.3j]),
]
a = g.group.inner_product(gradients, directions)


def shifted(h):
    return [g(v + h * dv) if isinstance(v, g.lattice) else v + h * dv for v, dv in zip(f.parameters(), directions)]


h = 1e-4
b = (-loss(shifted(2 * h)) + 8 * loss(shifted(h)) - 8 * loss(shifted(-h)) + loss(shifted(-2 * h))) / (
    12 * h
)
eps = abs(a - b) / abs(b)
g.message(f"Gradient: {a} vs {b}, rel {eps}; Im of the gradient of the real a: {gradients[0].imag}")
assert eps < 1e-8

# training through the optimizers: the functional's arguments are the live
# parameter list, so the optimizer updates the function's own storage
teacher = example(grid)
teacher.initialize(rng)
teacher["a"] = 1.3 + 0j
targets = [(xi, teacher([xi])[0]) for xi in [rng.cnormal(g.complex(grid)) for _ in range(2)]]

leaves = [rad.node(v) for v in f.parameters()]
c = sum(g.norm2(f([rad.node(xi, with_gradient=False)], leaves)[0] - ti) for xi, ti in targets)
cf = c.functional(*leaves)
initial = cf(f.parameters())
opt = g.algorithms.optimize.adam(maxiter=100, alpha=1e-1, eps=1e-15, log_functional_every=50)
opt(cf)(f.parameters(), f.parameters())
final = cf(f.parameters())
g.message(f"Training: loss {initial} -> {final}, a = {f['a']}")
assert final < 1e-2 * initial
assert f["a"].imag == 0.0

# the trained function evaluates with its updated storage
xi, ti = targets[0]
eps2 = g.norm2(f([xi])[0] - reference(f, xi)) / g.norm2(ti)
assert eps2 < 1e-28
