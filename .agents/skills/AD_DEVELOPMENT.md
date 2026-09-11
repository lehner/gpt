# GPT Development Guide — AD Framework

Notes for development sessions on the automatic-differentiation (AD) framework
and its QCD applications. Covers: using GPT, running the test suite, and how
the reverse-accumulation AD framework with lazy evaluation graphs works.

---

## 1. Environment and setup

- Repository layout:
  - `lib/gpt/`  — Python library (import as `gpt`, aliased `g` in tests)
  - `lib/cgpt/` — C++ core (Grid-based data parallelism: MPI, OpenMP, SIMD)
  - `tests/`    — test suite + `tests/run` driver
  - `applications/` — HMC and other physics drivers
- Machine (current dev box): **aarch64/ARM**, 4 cores, 15 GB RAM, **NEONv8** SIMD.
  Tests are written for small grids (2⁴–4⁴) to fit in memory.
- **Always source the build environment before running anything:**
  ```bash
  cd ~/GPT/gpt
  source lib/cgpt/build/source.sh
  ```
- Quick sanity check: `python3 -c "import gpt as g; print('ok')"`.
- **Stale bytecode bites:** after editing `lib/gpt`, clear caches before
  re-running tests, or you can get false regressions from old `.pyc` files:
  ```bash
  find lib/gpt tests -name "__pycache__" -type d -exec rm -rf {} +
  ```

## 2. Running the tests

- **Full suite** (the canonical invocation):
  ```bash
  bash tests/run
  ```
  With no arguments it defaults to `--mpi_split 1.1.1.1` (single-node). Passing
  a runner argument switches to that runner with your own extra args — for the
  default suite, do *not* pass a runner.
- **Individual test** (fast iteration):
  ```bash
  source lib/cgpt/build/source.sh
  python3 tests/ad/ad.py            # first-derivative AD checks
  python3 tests/ad/higher_order.py  # 2nd/3rd-order derivative checks
  python3 tests/qcd/gauge.py        # gauge action forces
  ```
- Test style: plain scripts, `g.message(...)` for progress, bare `assert`s.
  The workhorse is `differentiable_functional.assert_gradient_error`
  (`lib/gpt/core/group/differentiable_functional.py`), which checks that
  1) the functional is real, 2) the AD gradient agrees with a finite
  difference of the action along a random group-flow direction, and
  3) the force lives in the cartesian (Lie algebra) representation.
- **Adding a new test**: append its path to the list in `tests/run`.
- Known pre-existing failure on this box: `tests/qcd/coarsen.py` OOMs
  (15 GB RAM). Everything else should pass.
- Reproducibility: RNGs are seeded by name (`g.random("test")`), so results
  are bit-reproducible across runs on the same machine. When refactoring,
  compare reported numbers (relative errors) across the change — they should
  be bit-identical if behavior is preserved.

## 3. Using GPT (primer)

```python
import gpt as g

grid = g.grid([4, 4, 4, 4], g.double)     # Nx, Ny, Nz, Nt, precision
rng  = g.random("seed")

x = g.complex(grid); rng.cnormal(x)       # complex singlet lattice
r = g.real(grid);    rng.normal(r)

# SU(3) gauge field (list of 4 links), actions, forces
U   = g.qcd.gauge.random(grid, rng, scale=1.0)
act = g.qcd.gauge.action.iwasaki(6.0)
F   = act.gradient(U, U)                  # force, in cartesian (algebra) repr

# group operations
dA  = rng.normal_element(g.group.cartesian(U))   # random algebra element
Ue  = g.group.compose(g(eps * dA), U)            # group flow  (list-aware)
ip  = g.group.inner_product(dA, F)               # gauge contraction
gens = U[0].otype.cartesian().generators(grid.precision.complex_dtype)
```

Key concepts:

- **Lattices / tensors / blocks** are the data containers. `g(x)` evaluates a
  lazy expression `x` into a container (in place if possible).
- **Object types (otype)** tag containers with their algebraic structure:
  e.g. `ot_matrix_su_n_fundamental_group(3)` (3×3 unitary), its
  `cartesian()` → `..._fundamental_algebra(3)` (3×3 skew-Hermitian),
  adjoint representations, `ot_singlet`, etc. Many operations dispatch on
  the otype. The **cartesian** (Lie algebra) representation is where forces
  live; `infinitesimal_to_cartesian` / `cartesian_to_infinitesimal`
  convert between tangent-space perturbations and forces.
- **Expressions are lazy**: arithmetic on containers builds symbolic
  `g.expr` trees; evaluation happens at `g(x)`, `.real`, reductions, etc.
  Expressions do *not* know about AD nodes — mixing a plain container
  first-operand with a node second-operand (`plain * node`) can fail;
  prefer node-first order (`node * plain`) for node-aware dispatch.
- **Gauge actions** implement `__call__` (action value) and
  `gradient(fields, dfields)` (force). `act.transformed(diffeomorphism,
  indices, projection)` composes an action with a field transformation.
- `g.even_odd_projectors(grid)` gives the checkerboard masks `even`, `odd`.
- Finite differences of gauge actions along the group flow:
  `U(t) = [g.group.compose(g(t * dA[mu]), U[mu]) for mu in range(4)]`,
  central differences for 1st/2nd order, 4-point stencil for 3rd order.

## 4. The AD framework

Two independent mechanisms live under `lib/gpt/ad/`:

- **Forward AD** (`gpt.ad.forward`): differential algebra. `fad.series(field,
  On)` builds a Taylor series object; `eps = fad.infinitesimal("eps")`,
  `On = fad.landau(eps**2)` truncate at order. Coefficients are extracted as
  `field[eps]`, `field[1]`, .... Used where you need the *value and its
  directional derivatives simultaneously* (e.g. `adjoint_jacobian` in
  `lib/gpt/qcd/gauge/smear/differentiable.py`).
- **Reverse AD** (`gpt.ad.reverse`, the main one): node-based lazy graphs,
  detailed below.

### 4.1 Reverse AD: the lazy node graph

Core idea: wrap values in **nodes**; operations on nodes do *not* compute —
they build a **lazy compute graph**. Nothing is evaluated and no gradient is
computed until the root of the graph is **called**, at which point one pass
evaluates the graph (forward) and one pass accumulates all gradients
(backward).

```python
rad = g.ad.reverse

n = rad.node(x)                 # wrap a lattice/scalar/tensor
S = n**4 + 3.0 * n**2           # builds a lazy graph; nothing computed yet
S()                             # __call__: forward (eval) + backward (grad)
# now n.gradient holds dS/dx  (plain lattice, accumulated in place)
```

Mechanics (see `lib/gpt/ad/reverse/node.py`):

- `node(x, with_gradient=True)` creates a leaf. `with_gradient=False` marks
  a **constant** (no gradient accumulated for it; still participates in the
  graph as a fixed operand).
- Every node has:
  - `.value` — the value it holds (a container, or another node when nested);
  - `.gradient` — accumulated gradient (set by `backward`, reset by
    `zero_gradient()`);
  - `._forward` / `._backward` — set for *computed* nodes (built by
    `node_op`); `None` for leaves.
- `node_op(children, forward, backwards, container, tag)` is the constructor
  for computed nodes. `forward` is a zero-arg lambda producing the value from
  `value_of(child)`; `backwards` is one lambda per child producing the flow
  contribution. `container` fixes the result's type/otype (often via
  `get_mul_container` etc. in `util.py`).
- `S.__call__(with_gradients=True, initial_gradient=None)`:
  1. `traverse(nodes, self)` collects every node in the graph and records the
     dependency map;
  2. `forward(nodes)` evaluates each computed node once (values cached);
  3. `backward(nodes, ...)` walks the graph in reverse topological order,
     calling each node's `_backward` to scatter flows into
     `child.gradient`; leaf gradients are finally converted with
     `infinitesimal_to_cartesian(leaf.value, leaf.gradient)` so forces land
     in the cartesian representation.
  - `initial_gradient` seeds the root's gradient (default `1.0` for
    scalar-valued graphs; a *plain* lattice seed for field-valued graphs,
    e.g. `aUft[mu](initial_gradient=dU)` to apply the Jacobian to a
    direction `dU`).
- `value_of(x)` (in `ad/reverse/util.py`) evaluates a (possibly nested) node
  down to a plain value. To **release** a graph's memory after reading a
  result, repeatedly resolve `x = value_of(x)` while `is_node(x)`, then
  `g(x)` if it is an `expr`.

### 4.2 Conventions

- **Conjugate-linear (Wirtinger) convention:** the accumulated gradient is
  `conj(dS/dx)` — every backprop applies `g.adj` to its cofactor (visible in
  `__mul__`: `product(z.gradient, g.adj(value_of(y)))`). For real data or
  skew-Hermitian (effectively real) gauge directions this is invisible; for
  complex data it matters. When in doubt, test at `initial_gradient=1.0j`
  too, as `tests/ad/ad.py` does.
- **Contractions** (turning a gradient graph into a scalar so it can be
  evaluated):
  - scalar-valued node: `g.adj(na) * n.gradient` (node `__mul__` is
    adjoint-linear in its first argument);
  - lattice-valued node: `g.inner_product(direction, n.gradient)` (gradient
    in the linear second slot);
  - gauge-valued node: `g.group.inner_product(direction, n.gradient)`.
  - The contraction is symmetric in its arguments; a plain (non-node)
    operand is promoted to a constant node automatically.
- **Direction nodes** for contractions are created with
  `with_gradient=False`.

### 4.3 The functional (reusable-graph force) mechanism

Production actions use `node.functional(...)` →
`node_differentiable_functional` (`node.py`):

```python
n = rad.node(U)                       # or nested, see 4.4
f = S(n).functional(n)                # wraps the graph
val = f([U])                          # evaluate the value (re-evaluates graph)
gr  = f.gradient(U, U)                # force: plain lattices
```

`gradient()` overrides each leaf's value with a plain lattice and runs the
graph once, returning the **first** derivative only — regardless of node
depth. It is the force mechanism for actions, smears, etc., and is what
`differentiable_functional.assert_gradient_error` exercises. (2nd/3rd
derivatives use the nested-node mechanism below, *not* this one.)

### 4.4 Higher-order derivatives: nested nodes

A k-th derivative uses k nested wraps. Each **reverse pass deposits the next
derivative into the next-inner leaf's `.gradient`**:

```
n2 = rad.node(rad.node(x))
S(n2)()                          -> n2.gradient         graph for dS/dx
inner_product(a, n2.gradient)()  -> n2.value.gradient   d2S/dx2 * a  (plain)

n3 = rad.node(rad.node(rad.node(x)))
S(n3)()                          -> n3.gradient          dS/dx (node graph)
inner_product(a, n3.gradient)()  -> n3.value.gradient    d2S/dx2 * a (node graph)
inner_product(b, n3.value.gradient)()
                                 -> n3.value.value.gradient = d3S/dx3 * a * b
```

- Pass 1 (`S(n2)()`): the root gradient `n2.gradient` is itself a **lazy
  graph** (the 1st derivative as a function of the 1-deep nodes).
- Pass 2: evaluating a contraction of that graph runs a reverse pass *over
  the 1-deep graph*, depositing `d2S/dx2 · a` as a plain value in
  `n2.value.gradient` (the inner leaf).
- Pass 3 (for `n3`): the same, one level deeper.
- The core recursion is depth-general; no framework change is needed to go
  deeper, only memory (each level roughly doubles graph size).

Reference implementations: `tests/ad/higher_order.py` (scalars, lattices,
gauge HVP at 2-deep, gauge 3rd derivative at 3-deep, functional-force
mechanism at 2/3-deep) — this file is the executable specification of the
mechanism.

### 4.5 Where operations are implemented

- `ad/reverse/node.py` — `node`, `node_base.__mul__/__pow__/__truediv__/...`,
  `node_op`, `backward`, `functional`.
- `ad/reverse/transform.py` — transcendental transforms (sin, cos, ...) as
  node ops.
- `ad/reverse/util.py` — `nodify`, `product`, `value_of`, `is_node`,
  `value_depth`, `get_*_container`.
- `ad/reverse/foundation/` — the "foundation" layer: lattice-level
  implementations of ops the node layer dispatches to (trace/sum
  backprops, `where`, `astype`, group conversions), plus
  `foundation/matrix/exp.py` — **`matrix.exp` backward is pure Python**
  (Taylor/scaling-squaring built from div/add/multiply), not C++.
- "Foundation" is a per-class attribute (`g.lattice.foundation`, the
  node foundation, ...). Mixed-operand dispatch helpers (e.g.
  `_group_foundation` in `core/group/operation.py`) pick the operand whose
  foundation can handle both sides — a plain lattice foundation cannot
  evaluate a mixed lattice/node pair, so it defers to the node foundation
  (via `nodify`).

### 4.6 Pitfalls (learned the hard way)

- **otype is load-bearing.** Node containers carry the result otype. Leaf
  gradients are converted to cartesian by dispatching on the *gradient's*
  otype (`dsrc.otype.infinitesimal_to_cartesian(...)`). Any intermediate
  product that loses its group/algebra otype and collapses to bare
  `ot_matrix_color` will crash the leaf conversion with
  `AttributeError: 'ot_matrix_color' object has no attribute
  'infinitesimal_to_cartesian'`. Watch products of group elements with
  algebra elements / generators.
- **`plain * node` vs `node * plain`**: a plain lattice first-operand builds
  a symbolic `expr` that may not know how to handle a node
  (`Exception: Unknown type ...node_base`). Node-first invokes the
  adjoint-linear node `__mul__`.
- **Matrix × scalar node graphs**: a matrix-valued node graph multiplied by a
  scalar can break downstream trace contractions; route through
  `g.where(mask, x, zero)` for node graphs instead of in-place `x *= mask`.
- **Memory**: nested (2-deep/3-deep) graphs over gauge fields are expensive.
  Resolve results to plain lattices (`value_of` loop) to release graphs as
  soon as you're done reading them; use the smallest grid that exercises the
  code path; avoid holding multiple deep graphs alive at once.
- **Don't evaluate the same graph twice** expecting fresh state: backward
  accumulates into `.gradient` (call `zero_gradient()` first if reusing),
  and second reverse passes over an already-consumed 1-deep graph can
  produce nan. Build fresh nodes per pass when in doubt.
- **Initial gradients must match the node depth**: a 1-deep node's
  `initial_gradient` must be a *plain* lattice; a constant direction used in
  a *contraction* of a 2-deep root must be a `rad.node(dir,
  with_gradient=False)`.

## 5. File map (AD-relevant)

| Path | Role |
|---|---|
| `lib/gpt/ad/reverse/node.py` | node, node_op, forward/backward, functional |
| `lib/gpt/ad/reverse/util.py` | nodify, product, value_of, containers |
| `lib/gpt/ad/reverse/transform.py` | sin/cos/... node transforms |
| `lib/gpt/ad/reverse/foundation/` | lattice-level op backprops; `matrix/exp.py` |
| `lib/gpt/core/local_stencil/adjoint.py` | generic adjoint-stencil code derivation for compiled matrix stencils (see §5.2) |
| `lib/gpt/ad/reverse/stencil.py` | fused differentiable parallel transport (stage 0 of the differentiable-stencil work) |
| `tests/ad/stencil.py`, `tests/ad/stencil_adjoint.py` | stencil AD toy + random-code adjoint validation |
| `lib/gpt/ad/forward/` | series / Landau differential algebra |
| `lib/gpt/core/group/operation.py` | inner_product/compose dispatch |
| `lib/gpt/core/group/differentiable_functional.py` | action functional + `assert_gradient_error` |
| `lib/gpt/core/object_type/su_n.py` | SU(N) group/algebra otypes, generators, conversions |
| `lib/gpt/qcd/gauge/action/wilson.py` | reference gauge action |
| `lib/gpt/qcd/gauge/smear/differentiable.py` | `dft_diffeomorphism` (Jacobian machinery) |
| `tests/ad/ad.py` | 1st-derivative force checks (first-order tests belong here) |
| `tests/ad/higher_order.py` | 2nd/3rd-order nested-node reference (executable spec) |
| `tests/qcd/gauge.py` | gauge action + Hessian production conventions |
| `applications/hmc/hessian.py` | production Hessian/HVP convention |

## 6. Checklist for a new AD development session

1. `source lib/cgpt/build/source.sh`; clear `__pycache__` after edits.
2. Reproduce the baseline numbers of the affected tests *before* changing
   code (they are bit-reproducible).
3. For new derivative machinery: write a scalar/pointwise toy with a closed
   form first, then a gauge version, validating each pass against finite
   differences *before* touching production classes.
4. Add first-derivative checks to `tests/ad/ad.py`; 2nd/3rd-order checks to
   `tests/ad/higher_order.py` (or a dedicated test added to `tests/run`).
5. Run `bash tests/run` (no args) for the full suite; expect only the
   pre-existing `qcd/coarsen.py` OOM on the 15 GB box.
