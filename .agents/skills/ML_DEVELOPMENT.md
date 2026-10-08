# GPT Development Guide — ML Framework (`g.ml`)

Notes for development sessions on GPT's machine-learning framework: learnable
functions, their composition into networks, and training them with GPT's
optimizers.  `g.ml` is built on the reverse AD only.

**Read first:** `AD_DEVELOPMENT.md` (same directory) for the environment
(`source lib/cgpt/build/source.sh`, clearing `__pycache__`), the test suite,
and the AD itself (nodes, recorded higher derivatives, functionals, conventions,
pitfalls).  This file covers only what is specific to `g.ml`.

Goal of the framework: everything that is node-wrappable works in `g.ml`
(inputs, outputs, parameters), and the set of node-wrappable types is extended
over time.  Known AD gaps that limit `g.ml` are collected in
`~/GPT/TODOs/AD_gaps.md`.

---

## 1. Files and tests

| Path | Role |
|---|---|
| `lib/gpt/ml/function.py` | `g.ml.function` (base class), named storage (`_values`: value objects updated in place; numbers boxed as 0-d arrays) |
| `lib/gpt/ml/graph.py` | symbols, symbolic calls, `g.ml.pack`, `g.ml.composite`, `describe` |
| `lib/gpt/ml/monitor.py` | diagnostics: `snapshot`, `displacement`, `activity`, `gradient_noise` |
| `lib/gpt/ml/layer/basic.py` | `replicate`, `linear_combination`, `broadcast`, `polynomial` |
| `lib/gpt/ml/layer/local_covariant_matrix.py` | gauge-covariant residual block on matrix channels |
| `lib/gpt/ml/layer/mlp.py` | `matrix_invariants`, `mlp`, `matrix_words` (the covariant model f(P)) |
| `lib/gpt/ml/layer/word_sum.py` | site-local sums of products of matrix fields as one compiled stencil (used by `polynomial`) |
| `lib/gpt/ml/layer/util.py` | helpers shared by layers: `unit_scalar`, `embed` (c 1: a coefficient times the unit matrix, or a number as a field), `standardization` (calibrated mean / inverse std) |
| `tests/ml/function.py` | slots, names, storage, plain/node evaluation, type checks, gradients, training |
| `tests/ml/graph.py` | symbolic composition, ownership, sharing, nesting, `describe` |
| `tests/ml/local_covariant_matrix.py` | the covariant network: covariance, gradients, training, exact threshold solution |
| `tests/ml/monitor.py` | the diagnostics against hand-computed values |
| `tests/ml/layers.py` | `polynomial`, `matrix_invariants` / `mlp` / `matrix_words` (also with loops), as loop functions of transports |
| `tests/ml/loop_function.py` | a network as learnable loop function of two flow layers, trained on squared force contractions |
| `documentation/tutorials/advanced/ml-graphs.ipynb` | tutorial: functions, symbolic graphs, sharing, `describe`, covariance, training |
| `lib/gpt/ad/reverse/functional_node.py` | `g.ad.reverse.functional_node` (a functional as a node, first order) |
| `lib/gpt/core/group/differentiable_functional.py` | `g.group.directional_derivative` (and the functional base classes) |

The tutorial notebook is stored executed (outputs included, the import cell
without output).  After editing, re-run it in place with
`jupyter nbconvert --to notebook --execute --inplace <notebook>` from a shell
in which `lib/cgpt/build/source.sh` was sourced (the kernel inherits the
environment); the tutorials use the kernel name `python3-default`.

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
  namespace.  `f["c"]` is the list slot, `f["c.1"]` one element (a number
  for a number slot, else the live object); assignment is in place.
- **Storage.**  `f.parameters()` / `f.constants()` are the flat lists of the
  value objects in the order of `parameter_names()` / `constant_names()`.
  Values are updated in place and never replaced: numbers are stored as 0-d
  numpy arrays (of their dtype) and enter `evaluate` as numbers (plain:
  `.item()`; nodes: `g.ad.reverse.linear.element`, so a leaf of a 0-d array
  works to any order), arrays and fields are updated in place by
  assignments and by the optimizers (`set_element`).  The lists (`_values`)
  reject replacing an element (`x[i] @= y` reassigns the same object, which
  is allowed).  An optimizer given `f.parameters()` updates the function
  itself; to keep values, copy them (`g.ml.snapshot`).
- **Values** are numbers, numpy arrays, lattices or tensors (and lists of
  them).  Numpy-array parameters work through the site-constant linear maps
  of `g.ad.reverse` (`mlp`'s weights `W<l>` with `matrix_vector`, see
  `ad/reverse/linear.py`); other layers use list slots of numbers.  Long term
  `g.lattice` parameters are wanted.
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
   `evaluate` sees nodes only (one kind of value; mixed plain/node
   arithmetic would also work, see AD_DEVELOPMENT.md §4.6).
3. **Symbolic** (symbols as inputs, or a dict as parameters): the call is
   recorded, see §3.

`evaluate` must work on plain values and on nodes (also in recorded
passes, where its backward closures see nodes).  Plain
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

- **Calls** always return lists (consistent with concrete calls).  `label=`
  (optional) is a display string for `draw()`.  `name=`
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
  `parameters()`/`constants()` hold the same value objects as its functions
  (also through nested composites).  Since values are only updated in place,
  any number of composites may share functions (e.g. a training and an
  evaluation network), and `initialize`, assignments and optimizers act on
  the same values through any of them.  Lists of several functions'
  parameters (and other fields) are plain concatenations, e.g.
  `U + [rho] + net.parameters()`.
- **Lookup.**  `net["call.slot"]` delegates to the call's function (works for
  list slots, constants, and every call name of a shared function).
- **Nesting.**  A composite is a function: call it on symbols in another
  graph (`net([u1, u2], name="inner")` → names `inner.s1.a`).
- **`initialize(rng, scale=None)`** initializes each function once, each
  from its own stream `g.random(f"{rng.seed}/{prefix}")` (its first call's
  name): a function's initial values depend only on the seed and its name,
  not on the other functions (editing the graph keeps them).  `scale` is
  passed on to the functions if given.  **`calibrate(samples)`**
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
  `draw(ax=None)` draws the same graph with matplotlib (imported inside it
  only, an optional dependency; returns the figure): a layered layout with
  inputs left and outputs right in their signature order, calls in the column
  after their latest source, edges across several columns through waypoints
  (free lanes), rows ordered by alternating one-sided barycenter sweeps at the
  ports.  Boxes are sized to their text; edges leave at output ports and enter
  at slot ports (inputs, then connected parameters, dashed), labeled with the
  slot (and the output if a call has several).  Calls of a shared function
  share a border color (categorical slots 1-3, the all-pairs-safe ones) and
  name the call they share with; text uses ink colors only.  A box shows the
  call's `label` if it has one (`f([x], name="readout", label=r"$\oplus$")`:
  display only, matplotlib mathtext allowed, for figures in papers), else its
  name and function class.  `describe()` always shows names (they key the
  parameters and must stay identifiers).

### Serialization

```python
g.save(filename, net.state())
net.set_state(g.load(filename))
```

`state()` returns `{"parameters": {name: value}, "constants": {name: value},
"graph": describe()}` with the flat names of `parameter_names()` /
`constant_names()` and the live values (no copies), which `g.save` writes
directly (numbers, numpy arrays, tensors, lattices).  `set_state(state,
strict=True)` assigns by name (lattices and tensors in place, also from a
loaded lattice on a new grid object of the same layout; numbers and arrays
in place, numbers also from 0-d arrays; composites through their
functions).  Strict: the names
must agree exactly (missing and unknown names are listed); `strict=False`
assigns the names present in both.  Types are checked in any case (lattices
by grid description and otype, arrays by shape).  A different graph text
only warns (internal names may change).  Optimizer state (checkpoints) is
kept outside this.  `g.load(fn, grids={grid.describe(): grid})` puts loaded
lattices on the caller's grid.

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
- The optimizers (`adam`, `gradient_descent`, `non_linear_cg`, `lbfgs`, line search)
  and the node functional find parameters **by identity**
  (`g.util.index_by_identity`), so equal values are distinct parameters (the
  value objects of a function are distinct; the same object twice is still
  one parameter).
- `assert_gradient_error`, `g.group.cartesian`, `g.group.inner_product` and
  `rng.normal_element`/`uniform_element` handle numbers and numpy arrays
  (complex additive), so the generic gradient check covers mixed parameter
  lists.
- Training several runs from the same start: `ref = g.ml.snapshot(net)`
  (copies by name), restore with `net[name] = value` for each entry.  Do not
  keep `list(net.parameters())`: it holds the live objects.
- **`opt.on(x, dx=None)`** binds an optimizer to the fields `x` and the
  updated subset `dx` (default: all of `x`, the usual case for weights) and
  returns a run; `run(f)` iterates (`maxiter`, set at construction) with
  the functional `f`
  and keeps the optimizer's state (Adam's moments and step count) across
  calls, also when `f` changes from call to call: a cost drawn anew for
  every step is `run(make_cost(rng))` in a loop with an optimizer
  constructed with `maxiter=1`.  `opt(f)(x, dx)`
  is the same as `opt.on(x, dx)(f)` (repeated calls of one `opt(f)` keep
  its state as well).  A new `opt.on` / `opt(f)` starts a new state, i.e.
  restarts Adam's moments and bias correction (each restart begins with a
  full-size step).  Non-linear CG keeps its search direction within one call
  only (it belongs to `f`); gradient descent has no state; L-BFGS keeps its
  (s, y) pairs across the calls of a run.
- **`lbfgs`** (limited-memory BFGS in GPT operations: two-loop recursion with
  `g.group.inner_product`, strong-Wolfe line search, updates by
  `g.group.compose`) for deterministic costs, e.g. the force norm with
  v = F / |F|; it converges in far fewer evaluations than Adam.  The
  functional is evaluated at the fields as the optimizer sets them (in
  place), so a functional that reads a function's storage works.
  `failure_value` turns a `RuntimeError` at a trial point (a transport that
  cannot be inverted there) into a failed trial.
- Adam returns the last iterate, not the best; a too large `alpha` can end on
  an overshoot (5e-3 works for the covariant network).

### Diagnostics (`lib/gpt/ml/monitor.py`)

Whether a model trains, or still sits at its initialization:

- **`g.ml.gradient_noise(cost, fields, n, names=None)`**: `cost()` draws a
  new stochastic cost functional at the current fields; returns
  `{name: (|mean|, std, snr)}` of each field's gradient over n draws, with
  `snr = |mean|^2 / std^2` per draw.  At snr below about one a single draw's
  gradient sign is mostly noise, and Adam does a random walk of its step size
  (`alpha` per step, whatever the true gradient).  Fixes: average more draws
  per step, a smaller `alpha`, or a lower-variance estimator (for a force
  norm: the direction v = F / |F|, below).
- **`g.ml.snapshot(f)` / `g.ml.displacement(f, reference)`**: copies of the
  parameters by name, and `{name: (|p - p_ref|, relative)}` against them.
  Moves of the order of `alpha` per step mean the parameter is driven by
  noise or not at all.
- **`g.ml.activity(f, inputs)`**: on plain inputs, for each call of a
  composite `|y - x| / |x|` (its first output against its first input of
  the same type: the residual branch relative to the skip connection; None
  if there is none, e.g. `replicate`), and `""` for the function as a whole.
  A block can have moving parameters and still contribute nothing (e.g. a
  quadratic term with two small factors, gain times a readout weight).

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
- **Force norm without noise.**  For |F|^2 with F the force of an action,
  the cost `<v, F>^2` with the single fixed direction v = F / |F| (computed
  at the current parameters) has the value |F|^2 and the gradient of |F|^2
  (the dependence of v drops out to first order): an exact gradient at the
  cost of one random direction.  It is a different functional at every
  point, so check gradients with random directions.
- **Fields and weights in one list**: a plain concatenation, e.g.
  `U + net.parameters()`; optimizers and checks update the functions'
  values in place through it.  Numpy arithmetic on 0-d arrays gives numbers,
  so trial points of checks and line searches mix numbers and 0-d arrays;
  both are the same kind of field (`differentiable.assert_compatible`).

Numbers and numpy arrays work as fields of the action pipeline
(`transformed`, `added`, `directional_parallel_transport` with local and
generic Jacobians, log-det forces); all derivative fields are found by
identity.

- **Preimages as nodes**: `directional_parallel_transport.inv(fields)` accepts
  nodes (`g.ad.reverse.preimage`: plain inverse forward, backward from the
  transport's Jacobian by one small solve per step), so a latent field
  U(theta) = Phi_theta^-1(U0) of a fixed configuration U0 is an ordinary node
  value: `x = U0; for phi in steps: x = phi.inv(x + p)[0:4]` with the
  transport parameters `p` as nodes, then `functional_node(Q, x + p)`.  The
  gradient then includes the dependence of the latent field on the parameters
  (`applications/hmc/fthmc-learn.py --stage learn_rho`).  The inverse raises
  a `RuntimeError` (with the contraction rate of its fixed-point iteration)
  if it does not converge: training can drive a transport to where it is no
  longer invertible, so a training loop catches it and keeps its best point.

**Learnable loop function** of `directional_parallel_transport`
(`tests/ml/loop_function.py`): the network's weights are the transport
parameters, and the transport calls the network on its loop sums:

```python
dpt(U, description, mu, P0, P1, list(net.parameters()),
    loop_function=lambda sm, xp: net([sm], xp)[0])
```

With fixed loops (`loops=[g.path, ...]`, closed loops at x through no link
the step updates, e.g. the 2x1 rectangles around U_mu(x) for a checkerboard
P1) the loop function takes the list of loop fields as a third argument; a
network with inputs P and L (a list symbol; see `matrix_invariants` below and
`tests/ml/layers.py`):

```python
x, l = g.ml.symbols("P", "L")
inv = g.ml.layer.matrix_invariants(P, len(loops))
(I,) = inv([x, l], name="invariants")
(c,) = g.ml.layer.mlp(P, inv.n, g.ml.layer.matrix_words.n)([I], name="mlp")
(f,) = g.ml.layer.matrix_words(P)([x, c], name="words")
net = g.ml.pack(f=f).function()              # inputs [P, L]
net.calibrate([[P, L]])                      # L: a list of loop fields
dpt(U, description, mu, P0, P1, [rho] + list(net.parameters()), loops=loops,
    loop_function=lambda sm, xp, L: net([sm, L], xp[1:])[0])
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
  `rho_fn.parameters() + net.parameters()`).
- `polynomial(template, degree, scale=0.0)`: `x -> x + sum_{k=2}^{degree}
  c_k x^k` with numbers c, a covariant site-local polynomial as one compiled
  stencil (`word_sum`; coefficients enter as factor fields `c 1`).
- `matrix_invariants(template, n_loops=0)`, `mlp(template, n_in, n_out,
  width, depth, scale=0.0)`, `matrix_words(template)` (`layer/mlp.py`): the
  covariant model f(P) = P + sum_w c_w(I) w(P) with standardized invariants
  I (calibrated mean and inverse std, frozen), an MLP on them and covariant
  words in P, P^dag.  With `n_loops > 0` the invariants take a second input
  L (a list of n_loops matrix fields, the fixed loops of
  `directional_parallel_transport`) and add Re tr L_k / N, Im tr L_k / N per
  loop (`n = 5 + 2 n_loops`; for SU(3) loops tr L determines the eigenvalues,
  tr L L^dag = N is constant).  One layer rather than a separate loop layer:
  the mlp takes a single list input, and there is no concatenation layer.
  `loop_imag` (a flag or one per loop) drops Im tr L_k (e.g. for hermitian
  L_k); `mixed=True` adds Re tr P L_k / N, Im tr P L_k / N per loop.
  `matrix_words(template, n_loops, loop_adjoint)` takes L as a third input
  (`words([x, c, l])`) and adds the words linear in the loops, L_k, P L_k,
  L_k P (and L_k^dag if `loop_adjoint`), after those of P; its instance
  attribute `n` (and `names`) counts them (the class attribute
  `matrix_words.n` is the 6 words of P).
- `local_covariant_matrix(template, n_channels, gate=False, scale=0.1)`: a
  residual block on C channels of N x N matrix fields with
  `X_c(x) -> V(x) X_c(x) V(x)^dag`:
  `Y_c = sum_d (a_cd X_d + b_cd X_d^dag) + beta_c 1`, `Z_c = Y_c Y_{c+1}`,
  optional gate `relu(alpha_c (q_c - mean_c) inv_std_c + gamma_c)` with the
  invariant `q_c = tr(Y_c Y_c^dag)/N`, `X_c <- X_c + gain_c Z_c`.  Network: replicate
  -> L blocks -> linear_combination (see `tests/ml/local_covariant_matrix.py`).
  With a readout, the block's correction to the output is `w_c gain_c Z_c`,
  a product of two small initial factors (slow to start), and the readout's
  `w_c X_c ~ w_c x` duplicates any scale already in the input (e.g. rho of a
  loop function).  A single channel without readout (`replicate(P, 1)` ->
  block, output the list) has a correction linear in gain.

**Initialization convention.**  A layer is parameterized so that zero
(or a designated part of its parameters) gives the identity (or the neutral
element); `initialize(rng, scale=None)` draws around it with `scale` the
distance from the identity (None: the layer's default; 0: exactly the
identity).  Random scales follow the fan-in (sigma / sqrt(n) for a sum of n
terms).  Corrections that are quadratic in the weights get an output factor
initialized at `scale` (ReZero style, `local_covariant_matrix.gain`): at
`scale = 0` the block is the identity while the gradient w.r.t. the factor
is nonzero (inner weights at O(1)).  Stacked zero factors start learning
from the outside in (the readout first, then the blocks).

Writing a new layer:

1. Subclass `function` via `from gpt.ml.function import function` (not
   `g.ml.function`: `g.ml` does not exist yet while `gpt` imports its layers).
2. Declare typed slots from a template; build grid-dependent helpers (unit
   matrix, unit field) at construction as plain attributes, not slots.
3. Initialize near the identity following the convention above
   (`initialize(rng, scale=None)`; a function without parameters needs none).
4. Use only gauge-covariant operations if the layer claims covariance:
   scalar coefficients, products, adjoints, the identity, traces, and
   componentwise maps of invariants; `g.matrix.exp` commutes with conjugation.
5. Test: covariance (`f(V P V^dag) = V f(P) V^dag`, ~1e-30), node vs plain
   evaluation, `assert_gradient_error` over all parameters, a short training.

## 6. Pitfalls

- **Field + scalar** is not an expression (plain or node): multiply a unit
  field (`gamma * one`) or the identity matrix instead.
- **Numpy-array nodes** support element access (`d[0]`), `g.component.real`
  / `imag` and the linear maps of `ad/reverse/linear.py` (`matrix_vector`,
  `outer_sum`, `dagger`); no general whole-array arithmetic.
- **No `__rtruediv__` on nodes**: store reciprocals as constants (`inv_std`)
  instead of dividing by a node.
- **Componentwise node ops** are only relu, sin, cos, real, imag and
  `g.component.multiply` (primitives in `ad/reverse/transform.py`, so they
  work in recorded passes; relu's derivative drelu is a primitive without
  flow); `relu` on a complex z is z for Re z > 0, else a z.  Smooth gates
  need new node ops (as primitives, with a vjp written in primitives).
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
- `draw()` refinements: port order chosen to reduce crossings, dark mode,
  larger graphs (collapsing repeated blocks).
- General numpy-node arithmetic, parameters as `g.lattice`;
  `g.group.compose` for tensors; accelerator buffers as nodes.
- Symbol types (declared input types), symbolic element access of list
  symbols (`U[0]`).
- `functional_node` beyond first order; `directional_derivative` for
  non-abelian groups other than SU(N) fundamental.
- Fused (stencil) versions of layers for speed and a self-similar derivative
  tower (see AD_DEVELOPMENT.md §4.7).  Tried 2026-10-07 for `matrix_words`
  via `word_sum` (coefficients as `embed`ded factor fields c 1): at 16^4,
  14 words, plain evaluation -25% but value + gradient +12% (each
  coefficient becomes a matrix field, and its flow is traced back), so not
  adopted; a stencil taking the scalar coefficients directly would be needed.
