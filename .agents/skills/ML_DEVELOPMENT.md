# GPT Development Guide — ML Framework (`g.ml`)

Notes for development sessions on GPT's machine-learning framework: learnable
functions, their composition into networks, and training them with GPT's
optimizers.  `g.ml` is built on the reverse AD only.

**Read first:** `AD_DEVELOPMENT.md` (same directory) for the environment
(`source lib/cgpt/build/source.sh`, clearing `__pycache__`), the test suite,
and the AD itself (nodes, nested derivatives, functionals, conventions,
pitfalls).  This file covers only what is specific to `g.ml`.

Goal of the framework: everything that is node-wrappable works in `g.ml`
(inputs, outputs, parameters), and the set of node-wrappable types is extended
over time.  Known AD gaps that limit `g.ml` are collected in
`~/GPT/TODOs/AD_gaps.md`.

---

## 1. Files and tests

| Path | Role |
|---|---|
| `lib/gpt/ml/function.py` | `g.ml.function` (base class), named storage, `_storage_list` |
| `lib/gpt/ml/graph.py` | symbols, symbolic calls, `g.ml.pack`, `g.ml.composite`, `describe` |
| `lib/gpt/ml/layer/basic.py` | `replicate`, `linear_combination` |
| `lib/gpt/ml/layer/local_covariant_matrix.py` | gauge-covariant residual block on matrix channels |
| `tests/ml/function.py` | slots, names, storage, plain/node evaluation, type checks, gradients, training |
| `tests/ml/graph.py` | symbolic composition, ownership, sharing, nesting, `describe` |
| `tests/ml/local_covariant_matrix.py` | the covariant network: covariance, gradients, training, exact threshold solution |
| `tests/ml/loop_function.py` | a network as learnable loop function of two flow layers, trained on squared force contractions |
| `lib/gpt/ad/reverse/functional_node.py` | `g.ad.reverse.functional_node` (a functional as a node, first order) |
| `lib/gpt/core/group/differentiable_functional.py` | `g.group.directional_derivative` (and the functional base classes) |

Run a test with `python3 tests/ml/<name>.py`.  Long training runs are behind
`--stringent` (`g.default.has("--stringent")`): by default a test trains a few
steps and checks that the loss decreases, so all code is touched cheaply;
`--stringent` trains fully and asserts the reached losses.  New ML tests go
into `tests/ml/` and into the list in `tests/run`.

## 2. `g.ml.function`

A function maps named inputs to named outputs; it owns named parameters
(trained) and named constants (fixed, e.g. calibrated references).

```python
class scale(g.ml.function):
    def __init__(self, template):
        super().__init__(
            inputs=[("x", template)],           # name or (name, type)
            outputs=[("y", template)],
            parameters=[("a", 0j), ("c", [g.complex(grid), g.complex(grid)])],
            constants=[("s", 0.5)],
        )

    def initialize(self, rng):                  # separate from construction
        self["a"] = 0.8 + 0j
        rng.cnormal(self["c.0"])

    def calibrate(self, samples):               # optional, data-dependent constants
        ...

    def evaluate(self, inputs, parameters, constants):
        (x,) = inputs; a, c = parameters; (s,) = constants
        return [g(g.component.real(a) * x + s * c[0] * c[1] * x)]
```

- **Slots.**  Inputs and outputs are names or `(name, type)` with the type an
  AD container or a representative value (`None`: unchecked).  Parameters and
  constants are `(name, value)` with the allocated storage.  A list value is a
  list slot; its elements are named `c.0`, `c.1`, ... (a gauge field `U` is
  one slot `U.0`..`U.3`).
- **Names** are mandatory (they will key serialization): non-empty, no `.`
  (separator of list elements and composite levels), not pure integers
  (reserved for list elements), unique; parameters and constants share one
  namespace.  `f["c"]` is the list slot, `f["c.1"]` one element; assignment
  is in place (`@=`) for lattices/tensors, replacement for numbers/arrays.
- **Storage.**  `f.parameters()` / `f.constants()` are the live flat lists in
  the order of `parameter_names()` / `constant_names()`.  An optimizer given
  `f.parameters()` updates the function itself.
- **Values** are numbers, numpy arrays, lattices or tensors (and lists of
  them).  Long term numpy arrays and `g.lattice` parameters are wanted; today
  layers use list slots of numbers (numpy nodes support element access only).
- **Types select the update.**  There is no per-parameter type metadata: the
  optimizers update every parameter with `g.group.compose(step, x)`, which
  dispatches on the otype (additive groups add, U(1)/SU(N) exponentiate and
  multiply); numbers and numpy arrays are always complex additive.  So a
  general complex matrix parameter is `g.matrix_color_complex_additive`, not
  `g.mcolor` (an SU(N) element, whose gradient is projected onto the algebra).
- **Real parameters.**  Numbers are complex by default.  A parameter meant to
  be real is used as `g.component.real(p)` in `evaluate`: its gradient
  (dL/dRe + i dL/dIm) is then real and the additive update keeps a real value
  real.  `g.component.real`/`imag` are node ops for numbers, arrays and lattices.
- **`initialize(rng)`** is separate from construction (a network is built,
  then initialized or loaded).  **`calibrate(samples)`** sets data-dependent
  constants (`samples`: a list of plain input lists); default: nothing.

### Calling a function

`f(inputs, parameters=None, name=None)` has three modes:

1. **Plain** (no node among inputs/parameters): `evaluate` on plain values,
   no type checks (assumed checked when the training graph was built).
2. **Node** (any input or parameter is a node, e.g. leaves of a training
   graph passed as `parameters`): types are checked against the declarations
   (inputs, outputs) and the storage (parameters), using the nodes'
   `_container` (a computed node has no value before its graph runs); all
   plain inputs, parameters and constants are promoted to constant nodes, so
   `evaluate` sees nodes only and needs no node-first product ordering.
3. **Symbolic** (symbols as inputs, or a dict as parameters): the call is
   recorded, see §3.

`evaluate` must work on plain values and on nodes of any depth.  Plain
internal fields (e.g. a unit matrix built at construction) multiplied by
nodes are fine.

## 3. Composition: symbols, calls, `pack`, composites

```python
x1, x2 = g.ml.symbols("x1", "x2")
a, = sc([x1], name="s1")                       # returns a list of output symbols
p, q = sp([x2], parameters={"e": x1}, name="sp")   # parameter e computed from x1
y, = mx([a, p], parameters={"w": q}, name="mx")    # outputs of s1, sp -> mx
z, = sc([y], name="s2")                        # sc shared with s1
net = g.ml.pack(y=y, z=z).function()           # inputs [x1, x2] (creation order)
net = g.ml.pack(y=y, z=z).function(inputs=[x2, x1])   # or explicit
```

- **Calls** always return lists (consistent with concrete calls).  `name=`
  defaults to the class name; two reachable calls with the same name are an
  error in `function()` (name them explicitly).  Parameter connections name a
  list slot (`"c"`) or one element (`"c.1"`).
- **`pack`** takes the outputs as mandatory keyword arguments (symbols only).
  `function(inputs=None)`: default inputs are the free symbols the outputs
  depend on, in creation order; explicit inputs must be free symbols that are
  used, and every used symbol must be an input.
- **Ownership.**  A parameter slot of a function is a parameter of the
  composite if at least one call of that function leaves it unconnected (it
  is trained); a slot connected in every call is computed and not a composite
  parameter.  Composite names are `call.slot` with the name of the function's
  first call (`s1.a`, also used by `s2`); constants likewise (`mx.s`).
  Ownership is decided per composite.
- **Storage.**  Functions are the only owners of values.  A composite's
  `parameters()`/`constants()` are `_storage_list`s: fixed-length lists whose
  elements are read and written in the functions' storages (entries point
  through nested composites to the functions).  Hence any number of
  composites may share functions (e.g. a training and an evaluation network),
  and `initialize`, assignments and optimizers act on the same values through
  any of them.  `_storage_list` subclasses `list` (GPT checks
  `isinstance(x, list)`), keeps its own buffer empty and raises on every list
  method that would use it or change the length.
- **Lookup.**  `net["call.slot"]` delegates to the call's function (works for
  list slots, constants, and every call name of a shared function).
- **Nesting.**  A composite is a function: call it on symbols in another
  graph (`net([u1, u2], name="inner")` → names `inner.s1.a`).
- **`initialize(rng)`** initializes each function once.  **`calibrate(samples)`**
  replays the calls in order on the samples, calibrating each function (at its
  first call) on its call's inputs before evaluating it, so later functions
  see calibrated earlier ones.
- **Inspection.**  A symbol, a `pack` and a composite all show the subgraph
  reachable from their roots; only `pack` implements it (a symbol is a
  one-root pack, a composite keeps its pack, `net.graph()`).  `describe()`
  returns a text listing, one line per call:
  `mx2 = mix(u=s2.y, v=x1; c.1=sp.q)  parameters: mx.w, mx.c.0  constants: mx.s`
  (computed parameters after `;`, stored ones by composite name, so sharing is
  visible).  Nested composites are single lines (describe them directly).
  `draw(ax=None)` is planned (matplotlib imported inside it only, optional
  dependency) and raises `NotImplementedError` for now.

## 4. Training

```python
leaves = [rad.node(v) for v in net.parameters()]
loss = sum(g.norm2(net([rad.node(x, with_gradient=False)], leaves)[0] - t) for x, t in data)
cf = loss.functional(*leaves)                     # graph built once
cf.assert_gradient_error(rng, net.parameters(), net.parameters(), 1e-4, 1e-8)
g.algorithms.optimize.adam(maxiter=300, alpha=5e-3)(cf)(net.parameters(), net.parameters())
```

- Build the loss graph **once** and let the functional swap leaf values (graphs
  are reference cycles; see AD_DEVELOPMENT.md §4.6).  `sum(...)` works on
  nodes (`0 + node` is the node).
- The optimizers (`adam`, `gradient_descent`, `non_linear_cg`, line search)
  and the node functional find parameters **by identity**
  (`g.util.index_by_identity`), so equal values are distinct parameters.  The
  same object twice is still one parameter; note `complex(z)` returns `z`
  itself for a complex `z`.
- `assert_gradient_error`, `g.group.cartesian`, `g.group.inner_product` and
  `rng.normal_element`/`uniform_element` handle numbers and numpy arrays
  (complex additive), so the generic gradient check covers mixed parameter
  lists.
- Training several runs from the same start: save `list(net.parameters())`
  and write it back element-wise (numbers are replaced in the storage).
- Adam returns the last iterate, not the best; a too large `alpha` can end on
  an overshoot (5e-3 works for the covariant network).

### Actions and other functionals in a loss

Actions, flows and log-dets are `differentiable_functional`s with their own
(often non-node) gradient machinery.  They enter `g.ml` training through:

- **`g.ad.reverse.functional_node(S, fields)`**: the value `S(fields)` as a
  node (plain fields are constants), so it can be combined with other nodes.
  Its backward is `S.gradient` (converted to the infinitesimal convention),
  i.e. first order only.
- **`g.group.directional_derivative(S, v, along)`**: Q = <v, grad_along S>
  for a fixed cartesian direction v (one element per field index in
  `along`), itself a `differentiable_functional`.  Its gradient w.r.t. any
  field is d/de grad S(exp(e v) fields_along) at e = 0, a 4-point difference
  of `S.gradient` (symmetry of second derivatives); for SU(N) fields along v
  the non-commuting flows add -i [v, grad S].  Loss on force contractions:
  `sum(q * q for q in [functional_node(Q_i, U + leaves) ...])`.
- **`g.ml.fields(*lists)`**: one write-through list over several lists (e.g.
  `g.ml.fields(U, net.parameters())`), for functionals that take all fields
  as one list while an optimizer or a check must update the function's own
  storage.

Numbers and numpy arrays work as fields of the action pipeline
(`transformed`, `added`, `directional_parallel_transport` with local and
generic Jacobians, log-det forces); all derivative fields are found by
identity.

**Learnable loop function** of `directional_parallel_transport`
(`tests/ml/loop_function.py`): the network's weights are the transport
parameters, and the transport calls the network on its loop sums:

```python
dpt(U, description, mu, P0, P1, list(net.parameters()),
    loop_function=lambda sm, xp: net([sm], xp)[0])
```

## 5. Layers (`g.ml.layer`)

- `replicate(template, n)`: `x -> X = [x] * n`.
- `linear_combination(template, n, scale=0.01)`: `(x, X) -> x + sum_c w_c X_c`
  (a residual readout; small, nonzero initial `w` so earlier functions get
  gradients).
- `broadcast(template, value=0.0, real=False)`: no inputs, `y = value 1` (the
  unit element of the template's type; `Re(value)` with `real=True`).  Its
  backward sums the flow over the sites, so a global number can feed
  anything that takes a field, e.g. the field-valued description weight
  `rho` of `directional_parallel_transport` (`tests/ml/loop_function.py`:
  `rho_fn([], [rho_leaf])` in the loss graph, with the optimizer working on
  `g.ml.fields(rho_fn.parameters(), net.parameters())`).
- `local_covariant_matrix(template, n_channels, gate=False, scale=0.1)`: a
  residual block on C channels of N x N matrix fields with
  `X_c(x) -> V(x) X_c(x) V(x)^dag`:
  `Y_c = sum_d (a_cd X_d + b_cd X_d^dag) + beta_c 1`, `Z_c = Y_c Y_{c+1}`,
  optional gate `relu(alpha_c (q_c - mean_c) inv_std_c + gamma_c)` with the
  invariant `q_c = tr(Y_c Y_c^dag)/N`, `X_c <- X_c + Z_c`.  Network: replicate
  -> L blocks -> linear_combination (see `tests/ml/local_covariant_matrix.py`).

Writing a new layer:

1. Subclass `function` via `from gpt.ml.function import function` (not
   `g.ml.function`: `g.ml` does not exist yet while `gpt` imports its layers).
2. Declare typed slots from a template; build grid-dependent helpers (unit
   matrix, unit field) at construction as plain attributes, not slots.
3. Initialize near a known-good map (identity/residual), small random weights,
   nothing exactly zero that would block gradients.
4. Use only gauge-covariant operations if the layer claims covariance:
   scalar coefficients, products, adjoints, the identity, traces, and
   componentwise maps of invariants; `g.matrix.exp` commutes with conjugation.
5. Test: covariance (`f(V P V^dag) = V f(P) V^dag`, ~1e-30), node vs plain
   evaluation, `assert_gradient_error` over all parameters, a short training.

## 6. Pitfalls

- **Field + scalar** is not an expression (plain or node): multiply a unit
  field (`gamma * one`) or the identity matrix instead.
- **Numpy-array nodes** support element access only (`d[0]`), no whole-array
  arithmetic.
- **No `__rtruediv__` on nodes**: store reciprocals as constants (`inv_std`)
  instead of dividing by a node.
- **Componentwise node ops** are only relu, sin, cos, real, imag; `relu` on a
  complex z is z for Re z > 0, else a z.  Smooth gates need new node ops.
- **Gates need standardized invariants.**  Invariants such as tr(P P^dag)/N
  vary little between sites relative to their mean, so gate on
  `(q - mean)/std` with frozen calibrated references (frozen, not a live
  lattice average, to keep the function site-local).  Relu gates tend to
  collapse to all-on or all-off per channel.
- **Invertibility.**  As a `loop_function` of `directional_parallel_transport`
  the map must keep the Jacobian away from det = 0; strong functions or large
  rho make the log det ill-conditioned.  Training can drive the weights there
  (the log-det force diverges and the loss explodes): keep the step size
  small enough.
- **`functional.gradient`** leaves `with_gradient=False` on leaves not in
  `dfields`: do not reuse its graph for plain reverse passes afterwards.

## 7. Design principles and open items

Principles:
- Reverse AD only; every node-wrappable type should work as input, output
  and parameter.
- Functions are pure maps given their parameters, which are passed explicitly
  at call time (training graphs, nesting and computed parameters rely on it).
- Names are mandatory and hierarchical (they key serialization).
- The otype selects the update; numbers are complex, real intent is expressed
  with `g.component.real`.
- Construction, initialization and calibration are separate steps.
- Functions own all values; composites are views and may share functions.
- Type checks happen in node mode, inside the functions.

Open / planned:
- `draw()` with matplotlib (layered layout, boxes per call, solid input and
  dashed parameter edges, markers for shared functions).
- Parameters as numpy arrays (needs numpy-node arithmetic) and eventually as
  `g.lattice`; `g.group.compose` for tensors; accelerator buffers as nodes.
- Symbol types (declared input types), symbolic element access of list
  symbols (`U[0]`).
- Serialization (by the hierarchical names).
- `functional_node` beyond first order; `directional_derivative` for
  non-abelian groups other than SU(N) fundamental.
- Fused (stencil) versions of layers for speed and a self-similar derivative
  tower (see AD_DEVELOPMENT.md §4.7).
