#!/usr/bin/env python3
#
# Composing g.ml.functions: symbolic calls, pack, composites sharing
# functions, nesting, and inspection.
#
import gpt as g

rad = g.ad.reverse
rng = g.random("test")
grid = g.grid([4, 4, 4, 4], g.double)
field = g.complex(grid)


class scale(g.ml.function):
    # y = Re(a) x
    def __init__(self):
        super().__init__([("x", field)], [("y", field)], [("a", 0j)])

    def initialize(self, rng):
        self["a"] = complex(1.0 + 0.1 * rng.normal().real)

    def evaluate(self, inputs, parameters, constants):
        (x,) = inputs
        (a,) = parameters
        return [g(g.component.real(a) * x)]


class split(g.ml.function):
    # p = b x, q = e x x  (two outputs)
    def __init__(self):
        super().__init__(
            [("x", field)],
            [("p", field), ("q", field)],
            [("b", g.complex(grid)), ("e", g.complex(grid))],
        )

    def initialize(self, rng):
        rng.cnormal(self["b"], sigma=0.3)
        rng.cnormal(self["e"], sigma=0.3)

    def evaluate(self, inputs, parameters, constants):
        (x,) = inputs
        b, e = parameters
        return [g(b * x), g(e * x * x)]


class mix(g.ml.function):
    # y = w u + s c_0 v + c_1  (two inputs)
    def __init__(self):
        super().__init__(
            [("u", field), ("v", field)],
            [("y", field)],
            [("w", g.complex(grid)), ("c", [g.complex(grid), g.complex(grid)])],
            [("s", 0.5)],
        )

    def initialize(self, rng):
        for name in ["w", "c.0", "c.1"]:
            rng.cnormal(self[name], sigma=0.3)

    def evaluate(self, inputs, parameters, constants):
        u, v = inputs
        w, c = parameters
        (s,) = constants
        return [g(w * u + c[0] * v * s + c[1])]


def build():
    sc, sp, mx = scale(), split(), mix()
    x1, x2 = g.ml.symbols("x1", "x2")
    a, = sc([x1], name="s1")
    p, q = sp([x2], parameters={"e": x1}, name="sp")  # e is always computed
    y, = mx([a, p], parameters={"w": q}, name="mx")  # outputs of s1 and sp -> mx
    z, = sc([y], name="s2")  # sc shared with s1
    t, = mx([z, x1], parameters={"c.1": q}, name="mx2")  # mx shared, other connections
    symbols = dict(x1=x1, x2=x2, y=y, t=t, q=q)
    return g.ml.pack(y=y, t=t).function(), (sc, sp, mx), symbols


def reference(net, x1, x2):
    a, b, w, c0, c1 = net.parameters()
    s = net["mx.s"]
    ya = a.real * x1
    p = b * x2
    q = x1 * x2 * x2
    y = g(q * ya + c0 * p * s + c1)
    z = a.real * y
    t = g(w * z + c0 * x1 * s + q)
    return [y, t]


net, (sc, sp, mx), sym = build()
net.initialize(rng)
x1 = rng.cnormal(g.complex(grid))
x2 = rng.cnormal(g.complex(grid))

# names: owned slots only (sp.e is computed in every call), with the prefix
# of the first call of a shared function; inputs in creation order
assert net.input_names() == ["x1", "x2"] and net.output_names() == ["y", "t"]
assert net.parameter_names() == ["s1.a", "sp.b", "mx.w", "mx.c.0", "mx.c.1"], net.parameter_names()
assert net.constant_names() == ["mx.s"]
assert net["s2.a"] == net["s1.a"] and net["mx2.c.0"] is net["mx.c.0"] is mx["c.0"]
assert net["mx.c"][1] is net.parameters()[4]
assert [c.name for c in net.calls()] == ["s1", "sp", "mx", "s2", "mx2"]

# inspection: the composite, its pack and a single symbol describe their graphs
expected = """inputs: x1, x2
  s1  = scale(x=x1)                  parameters: s1.a
  sp  = split(x=x2; e=x1)            parameters: sp.b
  mx  = mix(u=s1.y, v=sp.p; w=sp.q)  parameters: mx.c.0, mx.c.1  constants: mx.s
  s2  = scale(x=mx.y)                parameters: s1.a
  mx2 = mix(u=s2.y, v=x1; c.1=sp.q)  parameters: mx.w, mx.c.0  constants: mx.s
outputs: y=mx.y, t=mx2.y"""
g.message(net.describe())
assert net.describe() == expected and net.graph().describe() == expected
assert sym["q"].describe() == """inputs: x1, x2
  sp = split(x=x2; e=x1)  parameters: sp.b
outputs: sp.q"""
try:
    net.draw()
    assert False
except NotImplementedError as e:
    g.message(f"draw: {e}")

# the composite holds no values: it reads and writes its functions' storage
sc["a"] = 0.7 + 0j
assert net.parameters()[0] == 0.7 and net["s1.a"] == 0.7
net["s2.a"] = 0.9 + 0j
assert sc["a"] == 0.9 and sc.parameters()[0] == 0.9
net.parameters()[0] = 0.8 + 0j
assert sc["a"] == 0.8
b = sp["b"]
net["sp.b"] = rng.cnormal(g.complex(grid))
assert sp["b"] is b and net.parameters()[1] is b
try:
    net.parameters().append(0.0)
    assert False
except TypeError as e:
    g.message(f"Rejected: {e}")
g.message("Names and shared storage: ok")

# default call names (class names) and explicit inputs
u, v = g.ml.symbols("u", "v")
f1, f2 = scale(), scale()
(a,) = f1([u])
(b,) = f2([v])
try:
    g.ml.pack(a=a, b=b).function()
    assert False
except ValueError as e:
    g.message(f"Rejected: {e}")
(b,) = f2([v], name="second")
h = g.ml.pack(a=a, b=b).function(inputs=[v, u])
assert h.input_names() == ["v", "u"] and h.parameter_names() == ["scale.a", "second.a"]
h.initialize(rng)
ha, hb = h([x2, x1])
assert g.norm2(ha - f1["a"].real * x1) == 0.0 and g.norm2(hb - f2["a"].real * x2) == 0.0

# errors
for call, expected in [
    (lambda: scale()([u], name="f.g"), ValueError),  # invalid call name
    (lambda: scale()([u, v]), ValueError),  # input count
    (lambda: scale()([u], parameters={"nope": v}), KeyError),
    (lambda: mix()([u, v], parameters={"c": u, "c.1": v}), ValueError),  # connected twice
    (lambda: scale()([x1], parameters={"a": u}), TypeError),  # symbols and values
    (lambda: scale()([x1], name="s"), ValueError),  # name of a concrete call
    (lambda: g.ml.pack(a=[a]), TypeError),  # a list instead of a symbol
    (lambda: g.ml.pack(a=a).function(inputs=[u, v]), ValueError),  # v not used
    (lambda: g.ml.pack(a=b).function(inputs=[]), ValueError),  # v used, not an input
    (lambda: g.ml.pack(a=a).function(inputs=[a]), TypeError),  # not a free symbol
]:
    try:
        call()
        assert False
    except expected as e:
        g.message(f"Rejected: {e}")

# plain evaluation
out = net([x1, x2])
ref = reference(net, x1, x2)
eps2 = sum(g.norm2(o - r) / g.norm2(r) for o, r in zip(out, ref))
g.message(f"Plain evaluation: {eps2}")
assert eps2 < 1e-28

# node evaluation
leaves = [rad.node(v) for v in net.parameters()]
nout = net([rad.node(x1), rad.node(x2)], leaves)
eps2 = sum(g.norm2(n(with_gradients=False) - o) / g.norm2(o) for n, o in zip(nout, out))
g.message(f"Node evaluation: {eps2}")
assert eps2 < 1e-28

# types of connected parameters are checked by the called function (with nodes)
(x,) = g.ml.symbols("x")
(y,) = scale()([x], parameters={"a": x})
bad = g.ml.pack(y=y).function()
try:
    bad([rad.node(x1)])
    assert False
except TypeError as e:
    g.message(f"Rejected: {e}")

# gradient w.r.t. all composite parameters
t1, t2 = rng.cnormal(g.complex(grid)), rng.cnormal(g.complex(grid))


def loss(parameters):
    y, t = net([x1, x2], parameters)
    return g.norm2(y - t1) + g.norm2(t - t2)


leaves = [rad.node(v) for v in net.parameters()]
loss(leaves)()
gradients = [n.gradient for n in leaves]
assert gradients[0].imag == 0.0
directions = [0.3 - 0.2j] + [rng.cnormal(g.complex(grid)) for _ in range(4)]
a = g.group.inner_product(gradients, directions)


def shifted(h):
    return [
        g(v + h * d) if isinstance(v, g.lattice) else v + h * d
        for v, d in zip(net.parameters(), directions)
    ]


h = 1e-4
b = (-loss(shifted(2 * h)) + 8 * loss(shifted(h)) - 8 * loss(shifted(-h)) + loss(shifted(-2 * h))) / (
    12 * h
)
eps = abs(a - b) / abs(b)
g.message(f"Gradient: {a} vs {b}, rel {eps}")
assert eps < 1e-8

# a second composite of the same functions: the output y alone (a subset of
# the calls), and a third with a new call that leaves mx's w unconnected
net_y = g.ml.pack(y=sym["y"]).function()
assert net_y.input_names() == ["x1", "x2"]
assert net_y.parameter_names() == ["s1.a", "sp.b", "mx.c.0", "mx.c.1"]
(r,) = mx([sym["x1"], sym["x2"]], name="m3")
net_r = g.ml.pack(r=r).function()
assert net_r.parameter_names() == ["m3.w", "m3.c.0", "m3.c.1"]
assert net_r["m3.w"] is net["mx.w"]

# training: the optimizer updates the functions' storage through net, and
# the other composites see the trained values
teacher, _, _ = build()
teacher.initialize(rng)
targets = [
    (xs, teacher(xs)) for xs in [[rng.cnormal(g.complex(grid)) for _ in range(2)] for _ in range(2)]
]
leaves = [rad.node(v) for v in net.parameters()]
c = sum(
    g.norm2(o - r)
    for xs, ts in targets
    for o, r in zip(net([rad.node(x, with_gradient=False) for x in xs], leaves), ts)
)
cf = c.functional(*leaves)
initial = cf(net.parameters())
opt = g.algorithms.optimize.adam(maxiter=100, alpha=5e-2, eps=1e-15, log_functional_every=50)
opt(cf)(net.parameters(), net.parameters())
final = cf(net.parameters())
g.message(f"Training: loss {initial} -> {final}")
assert final < 1e-2 * initial
assert sc["a"] == net.parameters()[0] == net_y.parameters()[0] and sc["a"].imag == 0.0
eps2 = g.norm2(net_y([x1, x2])[0] - net([x1, x2])[0])
assert eps2 == 0.0
r_ref = g(mx["w"] * x1 + mx["c.0"] * x2 * 0.5 + mx["c.1"])
assert g.norm2(net_r([x1, x2])[0] - r_ref) / g.norm2(r_ref) < 1e-28

# nesting: a composite is a function of another composite
u1, u2 = g.ml.symbols("x1", "x2")
y, t = net([u1, u2], name="inner")
outer_scale = scale()
(w,) = outer_scale([t], name="post")
outer = g.ml.pack(y=y, u=w).function(inputs=[u2, u1])
assert outer.describe() == """inputs: x2, x1
  inner = composite(x1=x1, x2=x2)  parameters: inner.s1.a, inner.sp.b, inner.mx.w, inner.mx.c.0, inner.mx.c.1  constants: inner.mx.s
  post  = scale(x=inner.t)         parameters: post.a
outputs: y=inner.y, u=post.y"""
outer_scale.initialize(rng)
assert outer.parameter_names() == [f"inner.{n}" for n in net.parameter_names()] + ["post.a"]
assert outer.constant_names() == ["inner.mx.s"]
sc["a"] = 0.6 + 0j
assert outer["inner.s1.a"] == 0.6 and outer.parameters()[0] == 0.6
outer.parameters()[0] = 0.65 + 0j
assert sc["a"] == 0.65 and net["s1.a"] == 0.65
y_o, u_o = outer([x2, x1])
y_r, t_r = reference(net, x1, x2)
eps2 = g.norm2(y_o - y_r) / g.norm2(y_r) + g.norm2(u_o - outer["post.a"].real * t_r) / g.norm2(u_o)
g.message(f"Nested evaluation: {eps2}")
assert eps2 < 1e-28

