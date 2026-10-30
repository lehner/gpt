# GPT Development Guide — AD Framework

Notes for development sessions on the automatic-differentiation (AD) framework
and its QCD applications. Covers: using GPT, running the test suite, and how
the reverse-accumulation AD framework with lazy evaluation graphs works.
The ML framework built on it (`g.ml`: learnable functions, composition,
training) has its own guide: `ML_DEVELOPMENT.md` (same directory).

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
- **Two builds exist side by side** (`lib/cgpt/make` names its output dir
  after the Grid build dir's basename):

  | Grid build | cgpt build | comms |
  |---|---|---|
  | `~/GPT/Grid/build` | `lib/cgpt/build` | none (single rank; default, sourced by `~/.bashrc`) |
  | `~/GPT/Grid/build-mpi` | `lib/cgpt/build-mpi` | MPI (MPICH, `mpi3`) |

  The system MPI is **MPICH** (`mpicxx`/`mpirun` → `*.mpich`), not OpenMPI.
  The MPI build was made with:
  ```bash
  cd ~/GPT/Grid && mkdir build-mpi && cd build-mpi
  ../configure --prefix=$HOME/GPT/local-mpi --enable-simd=NEONv8 \
      --enable-comms=mpi-auto CXX=mpicxx CXXFLAGS=-fPIC
  cd Grid && make -j4            # library only; no `make install` needed
  cd ~/GPT/gpt/lib/cgpt && ./make ~/GPT/Grid/build-mpi 2
  ```
  Running under MPI: preferably with `tests/run.mpi` (2 ranks, sources the
  MPI build, sets `OMP_NUM_THREADS=2`, passes the `--mpi` layouts):
  ```bash
  cd ~/GPT/gpt
  export PYTHONPATH=; bash tests/run.mpi python3 tests/ad/ad.py
  ```
  or by hand (reset `PYTHONPATH` first, since `~/.bashrc` already put the
  non-MPI build on it), **always with `OMP_NUM_THREADS` set so that ranks x
  threads does not exceed the 4 cores** -- each rank otherwise starts one
  OpenMP thread per core, and the overcommitted ranks run very slowly:
  ```bash
  cd ~/GPT/gpt
  export PYTHONPATH=; source lib/cgpt/build-mpi/source.sh
  OMP_NUM_THREADS=2 mpirun -np 2 python3 tests/core/core.py --mpi 1.1.1.2
  OMP_NUM_THREADS=1 mpirun -np 4 python3 tests/ad/ad.py --mpi 1.1.2.2
  ```
  The product of `--mpi` must equal `-np`.  Grid's stencil communication
  uses MPI also within a node by default (`--shm-mpi 1`); `--shm-mpi 0`
  switches to puts into the peers' shared-memory windows, so test
  communication changes in both modes. Each rank allocates a 1 GB comms
  buffer by default, which matters with 15 GB RAM at `-np 4`.
- **Compile times on this box** (4 cores, 15 GB, measured 2026-10-01):
  - Grid `configure`: ~1 min.
  - Grid library only (`make -j4` inside `<build>/Grid`): ~11.5 min. `-j4`
    is fine memory-wise. A top-level `make` also builds tests/benchmarks,
    which GPT does not need.
  - cgpt (`./make <grid-build> 2`, 176 sources): ~25 min at 2 jobs. Each
    file takes ~20–40 s. Do not raise the job count much: the default of 16
    gets the build OOM-killed.
  - A full rebuild of both from scratch therefore takes about 40 min.
  - Partial rebuilds: `./make` recompiles a source only if it is newer
    than its object (header changes are not tracked).  To iterate on one
    type, `touch lib/instantiate/lattice_double_iMColor3.cc` (and
    `..._single_...`, the only precision with SIMD lanes on NEON) and run
    `./make <grid-build> 2` (~20 s each).  This links with stale objects, so
    keep virtual interfaces and shared struct layouts unchanged meanwhile,
    and finish with a full rebuild (`touch lib/*.cc lib/instantiate/*.cc`).
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
  Expressions do *not* know about AD nodes: for an operand they do not
  know they return `NotImplemented`, so `plain * node` is handled by the
  node's reflected operator (see §4.6).
- **Gauge actions** implement `__call__` (action value) and
  `gradient(fields, dfields)` (force). `act.transformed(diffeomorphism,
  indices, projection)` composes an action with a field transformation
  (applied to the fields `indices`; the action reads the fields
  `projection`; both any subsets, gradients mapped by field).
  Its gradient differentiates the inner functional only where needed: at
  the outputs of the transformation that are not identity maps and at the
  requested fields.  A diffeomorphism declares its identity outputs with
  `identity_outputs` (positions within `indices`) and then accepts
  `jacobian(fields, fields_prime, dfields, inputs=)` (the positions whose
  gradient is needed); `directional_parallel_transport` does (every output
  but U_mu), so a force (links only) computes no parameter flows in any
  step.  Chain a flow with its log dets as one chain,
  `S = S.transformed(s_k) + ld_k` per step, not each log det transformed
  through all later steps (quadratic in the number of steps).
  `wilson` and `improved_with_rectangle` (iwasaki, symanzik, dbw2) take
  value and force from the AD stencil action in
  `qcd/gauge/action/staple_stencil.py` (see §4.7), which takes the links
  only (ValueError for further fields); their hand-written
  `staples()` / `staple()` remain for the heatbath.  `g.qcd.gauge.smear.local_stout`
  is a thin wrapper around `directional_parallel_transport`, and so is
  `g.qcd.gauge.smear.spectral_loop` (one loop, loop function sum_k c_k P^k,
  analytic log det).  Its `fused_action(inner)` is `inner.transformed(step) +
  step.action_log_det_jacobian()` with one reverse pass for the pull-back
  and the log-det force (`g.qcd.gauge.smear.latent_action(action, steps)`
  builds a chain with it: fields U + the learnable coefficients of the
  steps in list order, each step reads its own; other steps act on the
  links only; the action reads the links only).
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
  graph as a fixed operand). `name="U0"` names the leaf for printing.
- `print(y)` shows the graph's cost (`node_str.py`): the leaves, then one
  line per temporary (shared nodes, the root, long expressions hoisted;
  single-use nodes inlined) with `[k]` lattice ops per line, `used Nx`, and
  `RECOMPUTES tK` for structurally identical nodes built twice; then op
  counts per op/type and the peak live lattice values (forward only / kept
  for the backward).  `print(x.gradient)` after `backward(create_graph=True)`
  shows the cost of the recorded reverse pass.  Primitive nodes keep their
  static arguments as `._static` for this.
- Every node has:
  - `.value` — the value it holds: a plain value (lattice, tensor, number,
    numpy array, list of these) or a forward-AD series, never a node
    (checked on every write, `node._value_slot`; `rad.node(rad.node(x))`
    raises);
  - `.gradient` — accumulated gradient (set by `backward`, reset by
    `zero_gradient()`);
  - `._forward` / `._backward` — set for *computed* nodes (built by a
    primitive, see below); `None` for leaves.
- Every computed node is the node of a **primitive** (below): the node
  arithmetic (`*`, `+`, `-`, `/`, `**`, list-element access, the container
  conversion of a factor) in `node.py`, the foundation ops (adj, trace, sum,
  cshift, inner_product, projections, identity, astype, where), and the
  rest (`linear.py`, `transform.py`, exp, stencils, ...).  The container
  fixes the result's type/otype (often via `get_mul_container` etc. in
  `util.py`).
- `S()` (`__call__(with_gradients=True, initial_gradient=None,
  retain_values=False, with_value=True, create_graph=False, wrt=None)`) runs
  the forward and the reverse pass and returns the value;
  `S.backward(...)` is the same with `with_value=False` (returns None, or the
  gradients of `wrt`).  Use `.backward()` wherever the value of a reverse
  pass is not read (user convention, 2026-10-08).  The passes:
  1. `traverse(nodes, self)` collects every node in the graph and records the
     dependency map; `util.enter_pass` stamps them with the pass id (a pass
     nested in a running pass must not reach its nodes, §4.6);
  2. `forward(nodes)` evaluates each computed node once (values cached);
  3. `_reverse(nodes, ...)` walks the graph in reverse topological order,
     calling each node's `_backward` to scatter flows into
     `child.gradient`; leaf gradients are finally converted with
     `infinitesimal_to_cartesian(leaf.value, leaf.gradient)` so forces land
     in the cartesian representation.
  - `initial_gradient` seeds the root's gradient (default `1.0` for
    scalar-valued graphs; a *plain* lattice seed for field-valued graphs,
    e.g. `aUft[mu](initial_gradient=dU)` to apply the Jacobian to a
    direction `dU`).
  - `with_value=False` (with gradients, without `retain_values`): the root
    value is not needed, so only the values some backward reads are
    computed and `None` is returned (see §4.8); this is `S.backward()`.
  - `wrt=[leaves]`: the pass differentiates only these leaves (the others
    are constants for this pass and keep their gradients); per pass a node
    is active iff structurally differentiable and a selected leaf or with
    an active child, the structural flags are restored afterwards
    (`node._select`).  `wrt` only restricts: a leaf constructed with
    `with_gradient=False` (a constant) raises ValueError.
    `S.backward(wrt=[x, u])` returns `[x.gradient, u.gradient]`.
  - `create_graph=True`: records the reverse pass, see §4.4.
- `value_of(x)` (in `ad/reverse/util.py`) gives a node's value, evaluating a
  freed computed node in place; while a pass is recorded it returns the node
  itself (§4.4).

**Primitives** (`ad/reverse/primitive.py`, exported as
`g.ad.reverse.primitive` for user-defined operations: the "Defining new
operators" section of `documentation/tutorials/advanced/ad-reverse.ipynb`,
tested in `tests/ad/ad.py` and `tests/ad/higher_order.py`): an operation
closed under differentiation is declared once, by its plain implementation
and its vjp written in terms of primitives (itself or others):

```python
dagger = primitive("dagger", plain, container, vjp=lambda i, flow, W: dagger(flow), reads=((),))
```

Called on plain arguments it runs `plain`; with a node among the arguments
it returns a node whose forward is the kernel on the children's values and
whose backward is the vjp, which -- made of primitives -- runs plain kernels
on plain flows and, in a recorded pass, builds nodes of the same graph.  So
no op writes separate plain and recording code; the self-similar towers
(exp, stencils, the linear maps) are the primitive's own recursion.  Options: `joint_vjp(z, needed, *values)`
(all flows at once, e.g. one adjoint kernel for all inputs), `lift` (how an
argument becomes a child, e.g. `stack` for a list of fields), `reads` (the
vjp receives only the declared values, the others are None), static keyword
arguments (not differentiated, passed on to every level), `fwd` (the plain
value of a node plus a *residual* handed once to that node's vjp: the
exp tower's reset, `preimage`'s inverse, the packed input of `matrix_vector`), `order=1` (external
first-order nodes: `functional_node`, `preimage`, the Jacobian node of
`jacobian`), `jvp` (the tangent rule used by `g.ad.reverse.jacobian`,
usually built from `tangent.py`: `linear(op)`, `bilinear(op)`, `constant`),
`involution=True` (op(op(x)) is x: `adj`, `conj`), `fresh=True` (never shared,
for a node that marks a boundary by its identity: `jacobian.identity`).
**Shared nodes:** op on the same children (identity) with the same statics
returns the live node built before (if it holds no value and has the same
gradient flag), so a computation built twice -- by user code, or by the vjps
of a recorded pass, which build the same cofactor once per consumer -- is one
node (`primitive.share` / `.simplify` switch sharing / involutions off,
`primitive.stats` counts them; a retyped node is forgotten).  Code that
mutates a node a primitive returned would change every user of it; plain
arguments become new constants, so `0.5 * x` twice is not shared.  Study:
`~/GPT/TODOs/ad_hashcons_study/`.
A vjp returns a flow, None (no flow) or `flow.negative(r)` (-r, subtracted
without being built, also for a list flow).  A primitive's node never reads its
own value (`_reads_self = False`), so with `with_value=False` its forward is
skipped when nothing reads the value: state that a forward would refresh
must be refreshed in the vjp as well (the exp tower: see §4.5).

### 4.2 Conventions

- **Conjugate-linear (Wirtinger) convention:** the accumulated gradient is
  `conj(dS/dx)` — every backprop applies `g.adj` to its cofactor (visible in
  the vjp of `*` in `node.py`: `flow * g.adj(y)`). For real data or
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
n = rad.node(U)
f = S(n).functional(n)                # wraps the graph
val = f([U])                          # evaluate the value (re-evaluates graph)
gr  = f.gradient(U, U)                # force: plain lattices
```

`gradient()` overrides each leaf's value with a plain lattice and runs
`node.backward(wrt=[requested leaves])`, returning the gradient of the
graph's root. It is the force mechanism for actions, smears, etc., and is
what `differentiable_functional.assert_gradient_error` exercises.  The root
may itself be a contraction of a recorded gradient (§4.4), so the
functional of a Hessian bilinear form gives the 3rd derivative, and swapping
leaf values re-evaluates the recorded graph without re-recording.

### 4.4 Higher-order derivatives: recorded reverse passes

`S.backward(create_graph=True)` records the reverse pass: the backward
closures see the children *nodes* instead of their values (`util.record`;
`value_of` returns the node itself while recording), so the flows -- and the
leaf gradients -- are lazy nodes of the same graph, over the same leaves.  A
contraction of a recorded gradient is an ordinary scalar node; its reverse
pass gives the next derivative.  The order is chosen per pass:

```
n = rad.node(x)
S(n).backward(create_graph=True)                     -> n.gradient: graph for dS/dx
g.inner_product(a, n.gradient).backward()            -> n.gradient = d2S/dx2 a   (plain)

S(n).backward(create_graph=True)
g.inner_product(a, n.gradient).backward(create_graph=True)
g.inner_product(b, n.gradient).backward()            -> n.gradient = d3S/dx3 a b
```

- Directions and constants are plain values (or `rad.node(c,
  with_gradient=False)`); there is no depth matching.  Gauge fields contract
  with `g.group.inner_product` (applications/hmc/hessian.py).
- **Record once, contract many times**: the recorded gradient (keep a
  reference, `dS = [x.gradient for x in n]`; a later pass overwrites
  `x.gradient` but not the recorded node objects) can be contracted with
  any number of directions (HVPs, Lanczos), and swapping leaf values
  re-evaluates it (functionals, §4.3).
- **Values**: the recording reads no values.  `S(n)(create_graph=True)`
  runs a plain forward first (the returned value) whose intermediate values
  the reverse pass frees, so the next pass recomputes them: use
  `S(n).backward(create_graph=True)` (no forward at all) unless the value is
  needed (benchmark: +10-14% for an HVP otherwise).  `retain_values=True`
  keeps the forward values for the following passes over the recorded graph
  (the log-det force: one forward shared by 8 generators).  Only retain
  values while the leaves are unchanged.
- **Mixed derivatives**: differentiate the contraction w.r.t. other leaves
  (`d/dW <d, dS/dh>`); `wrt` restricts the recorded pass (e.g. the
  Jacobian block of U_mu only, `directional_parallel_transport`), the next
  pass differentiates all leaves.  Nodes built during a pass restricted by
  `wrt` (the recorded flows) take the *structural* flags of their children
  (`node._structural`), so the recording keeps the graph's dependencies.
- **Leaf conversion**: a recorded group-leaf gradient is converted
  (`infinitesimal_to_cartesian`) by node operations at the leaf, so the
  conversion is differentiated (it halves the derivative of the gradient
  graph: the factor 2 in the log-det forces).  A leaf created with
  `infinitesimal_to_cartesian=False` can convert its recorded gradient
  explicitly (`g.infinitesimal_to_cartesian(x, x.gradient)`, see
  applications/hmc/cdfthmc-learn.py).
- First-order-only primitives (`order=1`: `functional_node`, `preimage`,
  the Jacobian node of `jacobian`) raise in a recorded pass; a function of
  J (`log_det`, ...) is differentiated w.r.t. everything by one ordinary
  reverse pass.
- Forward mode composes with this as values: a leaf value may be a forward-AD
  series (reverse over forward, `tests/ad/ad.py`).

Reference implementations: `tests/ad/higher_order.py` (scalars, lattices,
gauge HVP, gauge 3rd derivative, the functional of a Hessian bilinear form,
node values rejected) -- the executable specification; `tests/ad/ad.py` (end:
`wrt`, mixed second derivatives under `wrt`).  History: until 2026-10-08
higher orders used nested nodes (`rad.node(rad.node(x))`, `.value.gradient`,
static depths); the recorded passes gave bit-identical results and replaced
them (`~/GPT/TODOs/ad_single_level_reverse.md`).

### 4.5 Where operations are implemented

(Module roles: see the file map, §5.)

- `ad/reverse/foundation/` — the "foundation" layer: the node versions of
  the core ops the node layer dispatches to, as primitives (adj, trace,
  sum, cshift, inner_product, `where`, `astype`, identity, componentwise
  maps, group conversions), plus
  `foundation/matrix/exp.py` — **`matrix.exp` on lattice nodes is a tower
  of fused kernels**: exp is D_0 of the family D_k(X; H_1..H_k) =
  d^k exp_X(H_1..H_k), whose reverse flows are again D's at X^dag
  (D_{k+1} into X, D_k into H_i), so the gradient of exp is exp.  Each
  plain D_k is one compiled local stencil (multi-dual scaling-squaring +
  Paterson-Stockmeyer Taylor).  All D_k of one tower (the user's exp node
  and the flows built from it, to any order) share a `_tower`: X is
  identified by identity, and its scaling (the norm bound, which is the same
  for X and X^dag) and the materialized X^dag are computed once.  The user's
  root D_0 resets the tower once per pass (in its plain forward, or -- if the
  forward was skipped, `with_value=False` -- in its vjp, signalled by the
  forward's residual), so a leaf updated in place is never served stale.
  The tower keeps no field alive beyond its pass (since 2026-10-17): X is
  referenced weakly (it is a node value, freed by the pass) and X^dag is
  released when the outermost pass ends (`util.release_at_pass_end`, see
  §4.6 "Pass-scoped caches"); nested passes inside a backward (the chunked
  tangent replays of a log-det force) still share it.  Before, every exp
  node of a cached graph (e.g. a `dft_diffeomorphism`'s `aUft`) held two
  fields between passes (2.5-3.7 GB for a round of smoothed flow steps at
  16^4; `~/GPT/TODOs/ad_memory_checkpoint_report.md`).
  Caveat for recorded graphs: the recorded D_k flows are evaluated in a
  later pass, possibly without the root's forward (the tower identifies X
  by object identity); swapping leaf values by assignment (functionals) is
  safe, but do not modify a leaf that feeds exp directly in place
  (`leaf.value @= ...`) between passes over a recorded graph.  Non-lattice
  (tensor/scalar) nodes use the forward AD's scaling-and-squaring Taylor
  series as a graph of node arithmetic.
- `ad/reverse/foundation/__init__.py` also holds single-node
  **projections**: `traceless_anti_hermitian` / `traceless_hermitian` (the
  `qcd.gauge.project` functions dispatch here for nodes).  They are
  self-adjoint w.r.t. Re tr(a^dag b), so the backward is the same projection
  (a projection node in a recorded pass).  The su(N) group conversions
  (`infinitesimal_to_cartesian`) are written with them.
- (Removed 2026-09-30: *expression nodes*, one node per sum of products
  with generated kernels; no net gain over the per-operation graph and
  shared subexpressions duplicated.  Last version: commit b7e27c86.)
- **Foundation dispatch.**  The core does not know about AD.  A core
  function on containers dispatches on its operand's class attribute
  `foundation` (`first.__class__.foundation.<op>(...)`), whose module
  implements the op for that class: `core/foundation/lattice/__init__.py`
  (`g.lattice`), `core/foundation/tensor.py` (`g.tensor`) and
  `ad/reverse/foundation/__init__.py` (`node_base`, the same names, as
  primitives).  Entry points: `g.cshift`, `g.copy`, `g.astype`, `g.where`,
  `g.identity` / `g.identity_constant`, `g.infinitesimal_to_cartesian` /
  `cartesian_to_infinitesimal`, `g.inner_product`, `g.norm2`, `g.sum`
  (`core/transform.py`), `g.adj`, `g.trace` (`core/operator/unary.py`),
  `g.component.*` (`component_simple_map`, `component_multiply`),
  `g.convert`, `local_stencil.matrix` (`fields[0].foundation.local_stencil`),
  `g.local_program` (the first field on a derived foundation, §4.11).
  **Mixed operands**: the first operand's foundation decides, except when
  it is the plain lattice foundation and the other operand's is a derived
  one (a node): the plain foundation cannot evaluate a mixed pair, so the
  node foundation takes it and promotes the plain operand to a constant
  (`nodify`); `_foundation` in `core/transform.py` (inner products) and
  `_group_foundation` in `core/group/operation.py` (also: a symbolic
  expression has no foundation and defers to the other operand).
  **Adding a node-capable core op**: the core function dispatches through
  the foundation; the plain body moves to the lattice (and tensor)
  foundation unchanged; the node foundation implements it as a primitive
  with its vjp in primitives (and `reads`, `jvp` where they apply).
  **Operators and expressions** do not use the foundations but Python's
  reflected operators: an expression (`core/expr.py`) and a
  `g.matrix_operator` return `NotImplemented` for an operand they do not
  know (`gpt.expr(x)` raises `NotImplementedError`, which
  `matrix_operator.__mul__` turns into `NotImplemented`), so `A * node`
  arrives at `node_base.__rmul__`.  The call `A(x)` dispatches on the
  foundation of x (`matrix_operator.__call__` -> `foundation.matrix_operator(A,
  x)`; the plain foundations and expressions run the plain body,
  `matrix_operator.evaluate`), so code written for plain fields and nodes
  alike calls operators directly.  For a node both forms (`__rmul__` too)
  build the primitive `_apply` (`ad/reverse/foundation/__init__.py`): forward
  `A(x)`, vjp `A.adj()` applied to the flow (again `_apply`, so any order),
  container from the operator's `vector_space` as `matrix_operator.evaluate`
  allocates it; `A(x)` and `A * x` are one shared node.  So any linear
  operator with an `adj_mat` (e.g. `g.fft()`, its `adj`, `inv`,
  `adj().inv()`) works on lattice nodes; its adjoint must be the true
  adjoint (the gradient is A^dag applied to the flow).  List nodes and
  Python lists of nodes are not supported there yet.
- **Blocking fine -> coarse** (`g.block.sublattice(fine_grid, factor,
  coarse_grid=None)`, `core/block/sublattice.py`; factor per direction,
  full grids, any otype): `restrict(offset)` y(c) = x(b c + o) and its
  adjoint `prolong(offset)` (zero fill) are cached `g.copy_plan`s, which
  pull each coarse site from the rank holding it (any rank layouts of the
  two grids); `sum` / `broadcast` are cgpt's rank-local `block_sum` /
  `block_embed` (a coarse grid in another rank layout goes through a
  repartition copy).  All four are `g.matrix_operator`s paired so that
  `R.adj()` is `P` (one object, so the nodes the vjps build are shared),
  hence any order on nodes (`A(x)` or `A * x`).
  `g.qcd.gauge.blocked_links(U, sublattice)` gives V_mu(c) = prod_{k < b_mu}
  U_mu(b c + k mu), typed like U: it picks the links before multiplying
  them on the coarse grid (no fine products or shifts; 10x faster than
  restricting U_mu cshift(U_mu) at 16^4, and the blocking costs < 1% of a
  coarse Wilson-flow chain).  Tests: plain in `tests/core/block.py`,
  gradient in `tests/ad/ad.py`, HVP in `tests/ad/higher_order.py` (end);
  benchmarks `~/GPT/TODOs/ad_blocking_study/`.
- Mixed-operand dispatch helpers for groups: `g.group.inner_product` and
  `g.group.compose` dispatch through `_group_foundation`
  (`group_inner_product`, `group_compose`); the plain foundations'
  `group_compose` is `operation.plain_compose` (the former body of compose,
  bit-identical), the node foundation's composes by node operations (same
  type: `otype.compose`, retyped by `astype`; an algebra element into an
  SU(N) group: `exp(i a) b`, without the plain path's defect projection,
  whose derivative on the group is the identity).

### 4.6 Pitfalls (learned the hard way)

- **otype is load-bearing.** Node containers carry the result otype. Leaf
  gradients are converted to cartesian by dispatching on the *gradient's*
  otype (`dsrc.otype.infinitesimal_to_cartesian(...)`). Any intermediate
  product that loses its group/algebra otype and collapses to bare
  `ot_matrix_color` will crash the leaf conversion with
  `AttributeError: 'ot_matrix_color' object has no attribute
  'infinitesimal_to_cartesian'`. Watch products of group elements with
  algebra elements / generators.  Likewise a tensor becomes a plain number
  (container `complex`) only if its otype has `cast_to_complex` (singlets,
  the additive groups); a U(1) element stays a one-component tensor, so its
  gradient is converted to the u(1) algebra (a number seed of such a root is
  promoted to a tensor, `node._reverse`).
- **Plain operands next to nodes**: `plain * node`, `plain + node`,
  `plain - node` work (and give the same as an explicit constant node): the
  core expression algebra (`core/expr.py`, `core/tensor.py`) returns
  `NotImplemented` for operands it does not know, so Python calls the
  node's reflected operator.  The same holds for numpy (`node_base` sets
  `__array_ufunc__ = None`; a numpy scalar or 0-d array on the left is taken
  as the Python number, `node._left_operand`) and for forward-AD series
  (a series combines with series, infinitesimals and plain values only).
  So vjps write plain operators (`flow * g.adj(y)`), plain or recorded.
  Keep it that way: the core does not know about AD; a new operand type
  gets its arithmetic through its own (reflected) operators.  (`plain /
  node` is not supported: no node `__rtruediv__`.)
- **Masks**: a matrix-valued node times a real 0/1 field is fine as a plain
  product (`g(sm * P1)`, as in `directional_parallel_transport._update`); the
  former `g.where(mask, x, zero)` workaround is no longer needed (its
  backward allocated a zero field per call).  But `0 * x` and `x * mask`
  stay nan where x is not finite (e.g. the inverse of a matrix that vanishes
  outside the mask): use `g.where` with an explicitly zeroed field there.
- **`g.identity_constant(x)`** returns an identity shared per (grid, otype,
  checkerboard) that must NOT be modified (kernel inputs, node
  values).  `g.identity(x)` returns a fresh field; some callers
  modify it (e.g. the plain exp Taylor fallback), so it must not be cached.
- **Caches must not capture call arguments**: state cached across calls
  (transports, constants, graphs) is built plain from (grid, otype), or
  derived inside each graph from that graph's own nodes
  (`g.identity_constant(Q)` is a constant node of Q's graph), never from the
  fields of the call that happened to come first: a node of another graph
  captured there pulls that graph into every later pass (re-evaluated, its
  flows reset).  Guard: a pass started while another pass runs (inside a
  backward) raises if it reaches any node of an enclosing graph
  (`util.enter_pass`: every pass stamps its nodes with its id; legitimate
  nested passes -- chunked tangent replays, checkpoint and recorded local
  passes, functional gradients -- build their own graphs on fresh leaves
  and share no node, constants included).
- **`g.mcolor` leaves are group elements**: their gradients are converted to
  the algebra (`infinitesimal_to_cartesian`).  Finite-difference checks of
  derivatives w.r.t. general complex matrices (e.g. an HVP) need leaves
  with `infinitesimal_to_cartesian=False`.
- **Stencil kinds** (`g.stencil.matrix`, `comm_type` of
  `g.local_stencil.matrix`): points on the axes only and no temporaries ->
  Grid's cartesian stencil (`comm_type=0`); any other points, or kernel-owned
  temporaries -> the **general stencil** (`comm_type=2`,
  `lib/cgpt/lib/foundation/general_stencil.h`): a lookup table per (point,
  site) into the local field (offset and SIMD permute) or into a comm buffer
  filled by a halo exchange, which uses Grid's `StencilSendToRecvFrom`
  interface (transfer buffers in Grid's shared-memory heap; every remote site
  is sent once per rank pair; all fields of a kernel call share one
  communication phase).  The general stencil supports full grids only: a
  non-axis matrix stencil (or one with temporaries) on a checkerboarded grid
  raises `NotImplementedError` (the former padded matrix stencil for them
  was removed 2026-10-07; nothing used it).  Matrix stencils need no
  `data_access_hints` (the method is a no-op); `g.stencil.matrix_vector`
  (Dirac-type operators, block maps) still runs on halo-padded fields
  (`g.padded_local_fields`) and needs its hints.  `comm_type=1` (no
  communication) is the plain local kernel (single point, padded fields).
- **Allocator effects in benchmarks**: a run that frees fields larger than a
  lattice (e.g. halo-padded copies) raises glibc's dynamic mmap threshold,
  so later lattices of the run are served from the heap without page
  faults.  Without that, a 16^4 color-matrix lattice (9.4 MB) is mmapped and
  unmapped per allocation: local_stout E4 looked 6% slower on the general
  stencil than on the former padded one although its stencil kernels got
  faster.  Compare such runs with
  `MALLOC_MMAP_THRESHOLD_=268435456 MALLOC_TRIM_THRESHOLD_=1073741824`.
- **Memory**: large fields and retained graphs live as long as a Python
  name refers to them, and recorded graphs over gauge fields are expensive.
  `del` each pass's graph (and recorded gradients) before building the next
  (e.g. in a loop of reverse passes);
  release large temporaries (ng x ng adjoint matrices are 7x a color matrix
  for SU(3)) right after their last use; avoid holding several recorded
  graphs alive at once; test on the smallest grid that exercises the code path.
  To find where memory goes: GPT books every live lattice in
  `sys.modules["gpt.core.lattice"].mem_book` (allocated in `lattice.__init__`,
  removed in `__del__`); a dict subclass that counts bytes on `__setitem__` /
  `__delitem__` gives exact live and peak lattice memory per call, and
  `--verbose lattice_creation` records each lattice's creation stack (group
  the survivors by it).  Compare with the resident set before / after
  `malloc_trim(0)` (ctypes) to separate freed heap memory kept by glibc and
  C++-side data from live fields; `gc.collect()` before / after shows
  reference cycles.  Scripts: `~/GPT/TODOs/ad_memory_checkpoint_study/`
  (`prof_mem.py`, `cycles.py`).  For deep chains (smoothing) use checkpoints
  (§4.9).
- **Re-running a graph**: each backward pass starts every gradient at
  `None` (an unbuilt zero, see `flow.py`), so gradients do not carry
  over between passes; read a leaf's `.gradient` before the next pass.  By
  default the backward frees forward values, so each call re-runs the
  forward.  `root(initial_gradient=..., retain_values=True)` keeps them: repeated reverse
  passes (one per seed) over unchanged leaves share one forward, and
  `root(with_gradients=False, retain_values=True)` returns the root value
  with all intermediates kept, so a seed built from it shares nodes with the
  following reverse pass (as `directional_parallel_transport` does).  Only
  retain values while the leaves are unchanged (`forward` reuses any value
  that is not None).  Without `retain_values` no value survives a pass, the
  root's included, so re-running a graph after changing leaf values is safe
  (since 2026-10-08; before, the root kept its value and a re-run returned
  it stale).
- **Stencil `accumulate` is a field index, not a flag**: with several
  targets in one fused stencil, each target's rewrites must accumulate
  into *its own* field index (target 1 uses `accumulate: 1`, not `0`).
- **Functional calls leave leaf values stale**: `node_differentiable_functional`
  `__call__`/`gradient` overwrite each leaf's `.value` with whatever fields
  they were handed, and `assert_gradient_error` ends on finite-difference
  *composed* lattices. Reusing those leaf nodes in a later section silently
  differentiates the perturbed fields (small, confusing reference
  mismatches). Re-wrap fresh `rad.node(...)` leaves per section.
- **Initial gradients**: a plain lattice for a field-valued root; in a
  recorded pass the seed may be a node (e.g. built on the root itself,
  `cartesian_to_infinitesimal(root, right)` in the log-det forces), and
  its dependencies are differentiated by the next pass.
- **Graph lifetime**: a graph is freed by reference counting as soon as no
  name refers to it (checked 2026-10-09 for products, exp, stencil actions
  and `functional_node`): a primitive's closures receive their node as an
  argument and never capture it.  Keep it that way: a closure or attribute
  that refers to its own node makes the graph a reference cycle, released
  only by Python's cyclic garbage collector, which counts objects, not
  bytes -- large fields pile up between collections (seen as +75 MB per
  call at 16^4).  The same holds for an object whose bound methods are a
  primitive's plain/vjp: it must not store that primitive (node ->
  primitive -> bound method -> object -> primitive is a cycle; it kept every
  per-call log-det graph of `jacobian._chunked` alive until a collection;
  fixed 2026-10-17: build the primitive in a method and return only the
  node, `_chunked.node()`).  One cycle exists by design: a leaf with a
  recorded gradient (leaf -> flow -> recorded graph -> leaf); drop the leaf,
  or reset `leaf.flow = None`, when done.  For a repeated operation still
  build the graph once and swap the leaf values (as `dft_diffeomorphism` and
  `directional_parallel_transport._local_vjp` do): it saves the graph
  construction.  A cached graph then must not keep fields of its last pass:
  reset its leaves' gradients after reading them (`leaf.gradient = None`;
  the caller's list keeps its own references), else every cached step of a
  flow holds its leaf gradients (4 fields at 16^4 = 38 MB per step).
- **Pass-scoped caches**: `node.__call__` runs inside `util.pass_scope`
  (a depth counter: the passes nested inside a backward, e.g. the chunked
  tangent replays, count as one).  `util.release_at_pass_end(obj)` calls
  `obj.release()` when the outermost pass ends.  Use it for any value a
  primitive caches beyond its node's value (the exp tower's X^dag), so that a
  cached graph keeps nothing alive between passes while all sharing within a
  pass is kept.
- **Leaves without flow**: in a plain pass a lattice leaf that received no
  flow (e.g. a link a local map does not read) gets the cartesian zero
  (`g.group.zero`) directly, without building and converting a zero field.
- **Flows are adopted, not copied**: the first plain contribution to a
  gradient is adopted as is (`flow.accumulate`), so the same field can be
  the gradient of several nodes (both children of an add receive
  `z.gradient`).  Ownership travels with the value (`flow.dense.owned`, per
  element for lists): an adopted gradient is copied before an in-place
  update, and leaf gradients are copied if still adopted when handed out.
  Consequences for new code: a vjp must return fields it does
  not reuse or overwrite later (no persistent scratch buffers as results),
  and code that writes into a node's gradient in place must call
  `own_gradient()` first.  `node.gradient` is a property
  (the flow's value; a list gradient is a fresh list per read, so assign
  `n.gradient = ...` rather than writing into `n.gradient[i]`).

### 4.7 Stencil nodes (`ad/reverse/foundation/stencil.py`)

A compiled matrix stencil called with node fields (`stencil(out_node,
*input_nodes)`, or `g.parallel_transport_matrix(...)(nodes)`) becomes one
computed node: the node of the stencil's primitive (`_stencil_op`, cached on
the stencil per output container and input count), installed into the
output node (`node_base._become`: the output node keeps its identity and
container, everything describing the computation is the primitive node's).  Its forward is the compiled kernel; its backward is the
**adjoint code** (`adjoint_code`), derived in closed form as stencils (the
product rule per factor, shifts negated/relativized, adjoint flags
adjusted), so the gradient of a stencil is a stencil and the tower is
self-similar to any order.  The adjoint kernels are compiled at the first
backward (cache `stencil._node_adj`, key (output count, flowed inputs)).

- **Regime**: only outputs and local temporaries are written; outputs are
  never read; the first write of an output does not read its old value
  (fresh, or adding an input).  **Local temporaries** are kernel-owned
  per-site fields, declared with `g.stencil.matrix(..., temporaries=[...])`;
  they are not passed by the caller, whose fields are the remaining ones in
  index order.  Rules: (R1) temporaries are written
  and read at the zero point only; (R2) an entry that reads a temporary reads
  *all* its factors at the zero point; temporaries are built from inputs only
  (no chains), all writes of a temporary precede its reads, and they are
  written fresh or accumulating into themselves.  (Scratch fields passed by
  the caller as constants are rejected.)
- **One derivation** for any number of temporaries: **stage A** (with the
  local temporaries, if any) recomputes the temporaries and runs the reverse
  sweep over the entries writing outputs; by R2 the flows of entries reading
  temporaries are local, so they go into the input slots and into each
  temporary's flow lambda_T (written to memory); **stage B** (temp-free, only
  with temporaries) pushes lambda_T, read at shifted points, through the
  temporary definitions into the inputs.  Only flowed inputs get slots, and
  only temporaries defined from a flowed input get a lambda_T.  Layouts:
  A = [slots][lambda_T][output flows][values of all passed fields][local T],
  B = [slots][lambda_T][values]; without temporaries A is the classic single
  adjoint kernel.  Plain flows: one fused run (B accumulating into A's
  slots); recorded: both stages are stencil nodes of the same kinds, so
  higher derivatives stay in stencils.
- Temporaries need a kernel without communication inside (the general
  stencil exchanges halos before the kernel; `comm_type` 1 or 2), never
  Grid's cartesian stencil.  Use them where a subproduct is **shared** (the
  up/down staples of plaquettes and rectangles, `staple_stencil_code`);
  without sharing they only add a stage barrier and the lambda_T memory
  traffic (the Wilson action is faster as a plain plaquette-loop stencil).

Plain-run optimizations:

- **Flowed inputs only**: a stencil node's adjoint (with or without local
  temporaries) computes the slots of the inputs that carry a gradient and are
  referenced (the cache
  key includes them); the entries of other slots are dropped and the slots
  renumbered, in the plain run and in the recorded adjoint nodes.  Constant
  inputs (e.g. coefficient fields of a pass that does not differentiate
  them) cost nothing in the backward.  (Tested in `tests/ad/stencil.py`, also a single-input stencil,
  whose recorded adjoint is a one-element list node.)
- **Seedless adjoints**: if the flow into a single-output stencil is exactly
  `c * identity` (see §4.8), the adjoint kernels are compiled once per c
  with the flow factor dropped and c (conj(c) for an adjointed read) folded
  into the weights (`seedless_code`).  For a traced loop sum this removes one
  of k matrix products per k-link entry (72 -> 48 for the Wilson force).
- **Common-subexpression elimination of the executed kernels**:
  `g.stencil.matrix(..., cse=...)` / `g.local_stencil.matrix(..., cse=...)`
  compiles an execution plan in which repeated adjacent factor pairs (a pair
  and its adjoint reversal share one) become per-site temporaries, possibly
  built from temporaries (`core/local_stencil/cse.py`, greedy Re-Pair).  The
  stencil object keeps its original `points`/`code`/`temporaries` (the
  executed plan is `executed`), so the AD derivation -- including the recorded
  tower's `matrix(compiled[0], ...)` -- still derives from uncombined codes,
  while every plain run at any level uses the plan.  Fields the kernel writes
  are combined only at the zero point and per write version (live reads).
  A plan is used only if it saves >= 15% of the products
  (`min_saving_default`): the kernels are dominated by factor fetches, so a
  temporary costs about what it saves (measured: >= 18% fewer products ->
  4-14% faster kernels, <= 12% -> neutral or slower).  Enabled for the AD
  adjoint kernels (switch: `cse` in `foundation/stencil.py`); cartesian-only
  kernels run without temporaries and are never combined.  Effect:
  fused-loop iwasaki force -13%, iwasaki HVP -3..-4%, 3rd derivative
  -2..-5%; the production forces and local_stout are below the threshold and
  bit-identical.

### 4.8 Value needs and structured flows

- **Which values a backward reads** is declared per node:
  `_reads_children` (None = all children, else per child i the child
  indices the flow into child i reads) and `_reads_self` (default True).
  A primitive's `reads=` sets both (a primitive never reads its own value).
  Declared by the node arithmetic, the foundation ops, `transform.py`,
  `linear.py` and the stencil nodes (the inputs, never the output); the
  matrix functions of `jacobian.py` and the first-order external nodes
  read all children.  `needed_values` walks the
  graph from the root and `forward` skips every computed node nothing
  needs (only with `with_value=False`).  **Safety net**: `value_of`
  evaluates a missing value on demand, so an undeclared read is still
  correct -- but it cascades (the recomputation needs its own inputs) and
  silently loses the saving.  So a backward must not call `value_of` on
  inputs it does not really need: trace/sum build their identity from the
  container (`_reduction_identity`).
- **Typed flows** (`ad/reverse/flow.py`): a node's gradient is held as
  `node.flow`, one of `None` (no flow yet), `dense(value, owned)`,
  `scaled_identity(c, identity)` or `flow_list(elements)`; `node.gradient`
  is its value.  Trace and sum pass down `scaled_identity(c)` when the flow
  they broadcast is exactly `c * identity` (a scalar flow, or a scaled
  identity of their own) into a plain lattice; it is built into a field only
  when someone reads `.gradient` (`flow.built`), and adds as a scaled
  identity to another scaled identity, as a field to anything else.
  Consumers test `flow.scale(node.flow)`: the stencil seedless adjoints fold
  c into their weights and pass a dummy for the flow field when the seedless
  kernel no longer reads it, so the c 1 field is never allocated (Wilson
  force at 16^4: -5%).  In a recorded pass a scaled identity with a plain
  c (the seed's path) is a constant; the flows into trace/sum are nodes
  otherwise (dense flows).  Consumers apply their plain-only shortcuts only
  for plain values.  (Possible optimisation: a scaled identity with a node
  c; a seedless stencil adjoint with c is c times the one with 1.)
- **Deferred reductions** (`flow.pending_sum`, plain passes only): a number
  used in a lattice operation is converted (`_convert` in `node.py`), whose
  vjp sums the lattice flow over the sites.  In a plain pass it keeps the
  rank-local sum (`rank_sum`, no communication); the number's flow is then
  base + (the sum over the ranks of partial).  `_reverse` sets aside the nodes
  whose flow is pending and every node below them, continues the sweep,
  reduces all pending sums of the pass in one global sum per grid
  (`flow.flush`, an array of the partials), then processes the nodes set
  aside in their order (repeated if these produce new pending sums, e.g. a
  number computed from a lattice and used again).  A spectral_loop step's
  pass: 3 global sums -> 1.  Results are those of the undeferred pass up to
  the association of several contributions to one number.  Each reverse
  pass keeps its own list of pending nodes (a pass nested in a backward
  reduces its own); reading a pending gradient reduces at once; an
  exception reduces nothing (a global sum is collective).  Recorded passes
  keep their sum nodes.  A tensor-valued sum is reduced at once.  Switch:
  `g.ad.reverse.flow.defer` (tested in `tests/ad/ad.py`, end).

### 4.9 Checkpoints (`ad/reverse/checkpoint.py`)

`outputs = g.ad.reverse.checkpoint(f, inputs)` is `f(inputs)` as one
node (gradient checkpointing).  `f` maps a list of fields to a list of fields
of any types (an input may be passed through; e.g. an MD step: links,
momenta, numbers -> links, momenta) and works on plain fields and on nodes;
plain inputs give `f(inputs)`.  The node's forward runs `f` on
plain values without a graph, so the outer graph keeps only the inputs (its
children); its backward re-runs `f` on fresh leaves and back-propagates the
flows through this local graph (contraction `real(sum_j <c_j, y_j>)`, as
`jacobian._chunked`), which is freed afterwards.  Cost: one extra forward of
`f` per backward.

- **Any order, self-similar**: the flow into the inputs is again a
  checkpoint, of `h(x, c)` = the vjp of `f`.  In a plain pass it is evaluated
  at once; in a recorded pass it is a lazy node of the recorded graph (no
  recording happens inside), so higher derivatives stay checkpointed.  Called
  on nodes (the leaves of the local pass of the next level), `h` builds `f`
  and records its local reverse pass over them; so every level is an
  ordinary local graph.  The local passes create gradient-carrying leaves and
  restrict themselves with `wrt`.
- `inline_nested=True`: checkpoints called on nodes while a local pass of
  this checkpoint runs are `f` inline (no node).  A nested checkpoint in a
  recorded local pass re-runs its `f` once per level of its own tower (its
  vjp's value, the replay of that vjp, its own backward), so a
  Hessian-vector product through a checkpointed gradient whose `f` holds
  checkpointed smoothing steps runs each step 1 plain + 3 node times; inline
  once (node).  Costs the memory of the whole local graph of `f`.  Used by
  `slow_sector_shift` (md_loss_multigrid): 16^4 pull-back 3.87 -> 2.61 s
  (4.1 -> 3.0 plain gradients of Psi), peak RSS 6.4 -> ~9 GB.
- Limits: an `order=1` primitive inside `f` (`linear_solve`) raises when the
  second derivative is computed (not in the recorded pass, which only builds
  the lazy node); no tangent rule (a checkpoint on a `g.ad.reverse.jacobian`
  path raises).  The outputs form one list node, whose container holds one
  container per element (`util.list_container`), so they may differ in
  type, as may the flows into the inputs (the outputs of the vjp
  checkpoint); tested with an MD step [U, P, beta] -> [U', P', x] to third
  order.
- Use: `frozen_generators.transverse_stout(..., checkpoint=k)` /
  `transverse_wilson_flow(..., checkpoint=k)` checkpoint every k smoothing
  steps (default `None`: unchanged).  16^4: the pull-back of a frozen-shift
  step with a 20-step smoothed generator +3.45 -> +0.88 GB peak lattices at
  +30% time (k = 1; larger k saved less: the checkpoints, nd links per
  segment, are cheaper than one step's graph).  Tested in
  `tests/ad/checkpoint.py` (values and gradients bit-identical, Hessian-vector
  products and third derivatives to rounding; `inline_nested`: the same, and
  the replay counts).

### 4.10 Differentiable symplectic integrators (`ad/reverse/symplectic.py`)

`g.algorithms.integrator.symplectic` updates fields in place (closures,
`@=`).  `g.ad.reverse.symplectic` has the same schemes as pure maps of a flat
list of fields, `fields_prime = md(fields, tau)`, on plain fields (a plain
run, bit-identical to the in-place integrators) or nodes (the trajectory as a
graph):

```python
sympl = g.ad.reverse.symplectic
iU, iP = [0, 1, 2, 3], [4, 5, 6, 7]                       # fields U + P + [rho]
ip = sympl.update_p(iP, sympl.gradient_finite_difference(S, 4), iU + [8])
iq = sympl.update_q(iU, lambda P: P, iP)                   # mass term
md = sympl.OMF2(N, ip, iq)                                 # leap_frog, OMF4, nesting
x = [rad.node(f) for f in U + P + [rho]]
E(md(x, tau)).backward()                                   # x[i].gradient
```

- `update_p(dst, force, src)` / `update_q`: fields[dst] <- g.group.compose(-/+
  eps force(fields[src]), fields[dst]); dst, src are indices.  An update is a
  `symplectic_base` whose op is a pure function, so
  `g.algorithms.integrator.symplectic.schemes` builds the step sequence and
  `integrator` (also a `symplectic_base`, nestable) runs it.  The module holds
  no node checks besides the checkpoint's local pass: plain/node dispatch is
  the foundations' and the primitives'.
- Parameters of the actions (beta, transformation parameters such as rho,
  network weights) are further fields: read by the force (in `src`), never
  updated; their flows from all force evaluations add up.  Real parameters
  enter the action as `g.component.real(p)`.
- Forces of an action S w.r.t. its first n fields (the rest: parameters):
  `gradient(S, n)`, for S accepting nodes: a checkpoint (§4.9), forward the
  plain `S.gradient`, backward through a recorded gradient of S on local
  leaves (converted by `infinitesimal_to_cartesian`); any order.
  `gradient_finite_difference(S, n, epsilon=1e-3)`, for any
  `differentiable_functional` whose gradient covers all its fields
  (transformed actions, log dets): an `order=1` primitive whose backward is
  the gradient of `g.group.directional_derivative(S, v, along)` with v the
  flows into the forces (symmetry of second derivatives; 4 full gradients of
  S per force, O(epsilon^4)).
- Graph memory: the fields at each force evaluation (one checkpoint or
  primitive node per force).
- Any order through `gradient(S, n)`, also with parameters (numbers) among
  the force inputs: the recorded vjp of the checkpoint is a checkpoint whose
  outputs are the flows into links and parameters (a list node of mixed
  element types).  Limits: `gradient(S, n)` inside a checkpoint with
  `inline_nested=True` raises (inline, its `f` runs on computed nodes, and
  its recorded gradient needs leaves).  No force-gradient or implicit
  updates; node compose only for additive groups and SU(N).
- Tested in `tests/ad/symplectic.py` (plain bit-identical to the in-place
  integrators, gradients against finite differences, reversibility, dE/dbeta
  and dE/drho, the two forces against each other, second order, also with
  beta among the differentiated fields).

### 4.11 Local programs (`core/local_program.py`, `ad/reverse/foundation/program.py`)

`g.local_program(function, inputs=[otype, ...], outputs=[otype, ...])` is a
site-local straight-line program of complex scalar operations (sums,
products, weights, conj; `real`/`imag` are built from them) on the
components of a few fields, run as ONE tensor-stencil kernel
(`g.stencil.tensor`, zero point only).  `function` receives symbolic inputs
(`local_program.matrix` for an n x n otype, a `scalar` for a singlet, a list
for a vector) and returns one symbolic value per output; `prog(*fields)`
returns the list of output fields.  Use it where a chain of many small
site-local ops on traces and singlets (each a cgpt.eval forward and several
backward, each a kernel launch on GPUs) would otherwise be a node graph:
`spectral_loop._log_det_terms` (271 -> 50 evaluations per step, the force of
one round 0.31 -> 0.19 s at 8^4; it forms P = C U_mu^dag and P' = C U_mu'^dag
itself), and the plain projections `traceless_hermitian` /
`traceless_anti_hermitian` and the SU(N) `infinitesimal_to_cartesian` (one
kernel each instead of 3-4 evaluations, for single-block N x N fields on full
grids; `qcd/gauge/project.py`, `_fused` / `_program`, otherwise the
expression code), so the projection nodes' forward and backward and every
leaf conversion of an SU(N) gradient are one kernel.

- **Tracer**: symbolic scalars are hash-consed (one node per distinct
  subexpression; constant folding, weights pulled to the front), so the
  program evaluates each subexpression once and only what the outputs depend
  on (a matrix product of which only the trace is used costs its diagonal).
  Serial numbers make the code deterministic for a deterministic build.
- **Lowering**: sums of single-use products become one `mov` and `inc`/`dec`
  per term (a single-use conj folds into the `_cc` instructions, which
  conjugate the first factor); weights need a `mul` instruction (the tensor
  stencil reads weights only there); at most two factors per instruction.
  Intermediates live in per-site registers (`vcomplex(30)` scratch fields,
  temporaries of the kernel, allocated by liveness, cached per grid); a unit
  field provides constants.
- **Mixed otypes**: one kernel reads fields of different otypes (matrices,
  singlets, the registers); the C++ opens each field's view with its own
  element count, the vector type is that of the first field, so all fields
  must share one grid and precision (asserted).  Every GPT lattice is
  stored as complex numbers (real ones too), so real masks enter directly.
- **Derived programs** (symbolic, cached on the program): `adjoint(needed)`
  (inputs + one flow per output -> the flows of the needed inputs, the
  conjugate-linear convention: flow of a in c = a b is conj(b) flow_c, of a
  in c = conj(a) conj(flow_c), of a in c = w a conj(w) flow_c), recomputing
  the forward values in the same kernel; `tangent(fiber, k)` (inputs + k
  tangents of the inputs fiber -> k tangents of the outputs, direction
  major, one kernel); `select(outputs)`.  They are programs again.
- **Node version**: with a node among the fields, one primitive node per
  group of outputs of one otype (a list node for several; outputs of two
  otypes give two nodes, although a list node may hold elements of several
  types, §4.9).  Its vjp
  calls the adjoint program (plain flows: its kernel, all needed flows at
  once even of different otypes; recorded: its nodes, whose vjp is the
  adjoint of the adjoint), its jvp the tangent program, so it is
  differentiable to any order and works inside `g.ad.reverse.jacobian`.
- Limits: site-local only (no shifts), no transcendental ops (logs, exp,
  inverses stay node ops outside), complex weights only.  Coefficients that
  are learnable numbers enter as fields c 1 (one lattice sum per coefficient
  in the backward, not one per product with a number).
- Tested in `tests/ad/local_program.py` (plain values, gradients against the
  node graph of the same formula, complex seed, wrt, second and third
  derivatives, two output otypes, a program over three register fields, the
  tangent program in `jacobian` and its log-det force).

## 5. File map (AD-relevant)

| Path | Role |
|---|---|
| `lib/gpt/ad/reverse/node.py` | `node`, `node_base` (`__mul__`/`__pow__`/`__truediv__`/... as primitives; `_left_operand`: numpy scalars on the left; `__rmul__` of a `g.matrix_operator`: the foundation's `_apply`, §4.5), forward/backward, `functional`, `_become` (an output node takes over a computed node) |
| `lib/gpt/ad/reverse/node_str.py` | `str(node)`: the graph as a listing of temporaries with lattice-op costs, sharing, recomputed subexpressions and memory peaks (§4.1) |
| `lib/gpt/ad/reverse/primitive.py` | `primitive`: an op from its plain implementation and its vjp in primitives (plain/recorded dispatch, joint vjps, residuals, first-order ops, shared nodes, involutions), §4.1 |
| `lib/gpt/ad/reverse/tangent.py` | shared tangent rules (jvp) of the primitives: `linear`, `bilinear`, `constant`, `count`, `total` (used by `g.ad.reverse.jacobian`) |
| `lib/gpt/ad/reverse/flow.py` | typed flows (`dense`, `scaled_identity`, `flow_list`, `pending_sum`: deferred reductions, `flush`), `negative`, `accumulate`, `accum`, §4.8 |
| `lib/gpt/ad/reverse/util.py` | `constant` (a plain value as a constant node), `nodify`, `value_of`, `record`/`recording` (recorded passes), `pass_scope` / `release_at_pass_end` (pass-scoped caches, §4.6), `is_node`, containers (`get_container`, `get_*_container`, `list_container`: one container per element, `elements()`, `element()` the common container of a uniform list (TypeError otherwise), `same_container`: a copy of the first argument's container, the rule of every type-preserving primitive, since a node never shares its container with a child; a container's str is cached, so its tag changes only through `set_otype`; `otype_compatible` is memoized by otype name), local passes (`contraction`: real(sum_j <c_j, y_j>) whose reverse pass is a vjp; `recorded_gradients`: the recorded gradients of leaves, lazy zeros for none, flows reset) |
| `lib/gpt/ad/reverse/transform.py` | componentwise node ops: relu, drelu, sin, cos, real, imag, conj, log, pow (`g.component.inv` is pow(-1)), `multiply` (the foundation's `component_simple_map` / `component_multiply` dispatch here) |
| `lib/gpt/ad/reverse/functional_node.py` | a `differentiable_functional` as a node (first order, a joint-vjp primitive; used by `g.ml` losses) |
| `lib/gpt/ad/reverse/linear.py` | site-constant linear maps on lists of scalar fields (`stack`, `matrix_vector`, `outer_sum`, `dagger`; one gemm over the sites; packing costs more than the gemm, so the plain backward of `matrix_vector` packs the flow once for both flows, reads the h packed by the forward (a `fwd` residual) and computes flows only for the entries of h with a gradient, not a constant bias: -11% for the loop-function network value + gradient at 16^4), used by `g.ml.layer.mlp`; array element access `element` / `scatter` (each other's vjp; node `__getitem__` of arrays, the unboxing of `g.ml` numbers) |
| `lib/gpt/ad/reverse/jacobian.py` | `g.ad.reverse.jacobian(y, x, chunk=None)`: the site-diagonal Jacobian J(s) = dy(s)/dx(s) as a node (forward tangents replayed by the nodes' `jvp`, batched over the generators; `chunk`: fewer at a time), `identity` (x as a new node: the boundary of a local map), self-similar `inv`, `det`, `log_det` of J (tests in `tests/ad/jacobian.py`) |
| `lib/gpt/ad/reverse/checkpoint.py` | `g.ad.reverse.checkpoint(f, inputs, containers, inline_nested)`: gradient checkpointing, any order (§4.9) |
| `lib/gpt/ad/reverse/symplectic.py` | `g.ad.reverse.symplectic`: differentiable symplectic integrators `md(fields, tau)` (`update_p`, `update_q`, `leap_frog`, `OMF2`, `OMF4`) and the forces `gradient` (checkpointed recorded gradient) and `gradient_finite_difference` (first order, any functional) (§4.10; tests in `tests/ad/symplectic.py`) |
| `lib/gpt/ad/reverse/linear_solve.py` | `g.ad.reverse.linear_solve(op, rhs, solver, parameters=[], adj_op=None)`: the solution of a function op linear in x as a node (first order; backward: one adjoint solve, parameter flows by a local reverse pass of op) |
| `lib/gpt/ad/reverse/preimage.py` | the preimage x = phi^-1(y) of a diffeomorphism as nodes (first order; backward: solve J_xx^T lambda = c with `dfm.jacobian`, flows lambda and -(dphi/d others)^T lambda); `directional_parallel_transport.inv` accepts nodes through it |
| `lib/gpt/ad/reverse/foundation/` | the node versions of the core ops as primitives (also `matrix_operator`: `_apply`, a `g.matrix_operator` on a node, §4.5), projection nodes, group conversions; `matrix/exp.py` (exp tower) |
| `lib/gpt/core/local_program.py` | `g.local_program`: site-local straight-line programs on components as one tensor-stencil kernel (tracer, lowering with registers, derived `adjoint` / `tangent` / `select` programs), §4.11 |
| `lib/gpt/ad/reverse/foundation/program.py` | node version of `g.local_program` (one primitive per output otype; vjp and jvp as derived programs) (§4.11; tests in `tests/ad/local_program.py`) |
| `lib/gpt/ad/reverse/foundation/stencil.py` | node foundation for compiled matrix stencils (§4.7): adjoint derivation (`adjoint_code`, with or without local temporaries), multi-output list nodes, local temporaries, seedless adjoints |
| `lib/gpt/core/stencil/matrix.py`, `lib/gpt/core/local_stencil/matrix.py` | compiled matrix stencils (kind selection, `comm_type`; full grids for non-axis points); `temporaries=`; `cse=` |
| `lib/cgpt/lib/foundation/general_stencil.h` | general stencil: geometry (lookup table, halo transfer plan), per-field halo, batched exchange via Grid's `StencilSendToRecvFrom`, manager (field point sets); used by `stencil/matrix.h` (`comm_type == 2`, kernel loops in `stencil/matrix_loops.h`) |
| `lib/gpt/core/local_stencil/cse.py` | common-subexpression elimination of a kernel's execution plan (tested in `tests/core/stencil.py`) |
| `tests/ad/stencil.py` | stencil AD: fused two-output stencil, path-based stencils, local temporaries (staple action vs cshift graph) at 1st/2nd/3rd order, `with_value=False` |
| `lib/gpt/qcd/gauge/action/staple_stencil.py` | gauge action value/force as AD stencils (plaquette loop, or staples as local temporaries with rectangles) |
| `lib/gpt/qcd/gauge/smear/directional_parallel_transport.py` | checkerboarded smearing (behind `local_stout`): local jacobian/VJP via the staple (`g.staple_description`), the per-site Jacobian block (`jacobian_matrix`, optionally at a prescribed staple and loops), log-det and its force (also weighted per site), `inv` (`g.algorithms.nonlinear.fixed_point` with site-local Newton steps; `inverse_history`).  `loops=[g.path, ...]`: fixed closed loops at x as further inputs `f(sm, xparams, L)` of the loop function (one fused stencil, `_loop_transport`); the constructor rejects (ValueError) a loop through U_mu(x + s) if P1(x) P1(x + s) != 0 for some x, `inv` re-checks numerically.  Like the staple they are constants of the local map; their flows are pushed back through their own stencil (one list-node reverse pass; tests in `tests/qcd/flows_loops.py`) |
| `lib/gpt/qcd/gauge/smear/spectral_loop.py` | the spectral single-loop step U_mu' = exp(TA(P1 F(P))) U_mu, F(P) = sum_k c_k P^k, P = T0 U_mu^dag (SU(3)): map, `inv` and jacobian through a single-loop `directional_parallel_transport`; coefficients fixed (`{k: c_k}`) or learnable fields (`{k, ...}`: fields U + [c_k], plain or nodes); analytic log det as one `g.local_program` of (C, U_mu, U_mu', P1, coefficients) (P and P', power sums of tr P, tr B, tr B^2, shift-invariant discriminant for numerical stability) and three logs, any order; learnable number coefficients enter node graphs as fields c_k 1 (`_coefficient_fields`, shared by F and the log det); `fused_action`, `latent_action` (tests in `tests/qcd/spectral_loop.py`) |
| `lib/gpt/ad/forward/` | series / Landau differential algebra |
| `lib/gpt/core/block/sublattice.py`, `lib/gpt/qcd/gauge/block.py` | `g.block.sublattice`: restrict / prolong (copy plans) and block sum / broadcast between a fine grid and its blocks as paired `g.matrix_operator`s; `g.qcd.gauge.blocked_links` (§4.5) |
| `lib/gpt/core/group/operation.py` | inner_product/compose dispatch (`group_inner_product`, `group_compose` of the foundations; `plain_compose`), `zero` (cartesian zero) |
| `lib/gpt/algorithms/nonlinear/fixed_point.py` | `g.algorithms.nonlinear.fixed_point`: iteration control of x = Phi(x) (history, contraction rate, switch to an accelerated step, failure), `fixed_point.newton(residual, solve)` Newton steps; used by `directional_parallel_transport.inv` |
| `lib/gpt/core/group/algebra_kernels.py` | `g.group.algebra_kernels`: site-local kernels between SU(N) algebra fields and adjoint matrices (coordinates as rows, combinations of generators) |
| `lib/gpt/core/parallel_transport/matrix.py` | weighted parallel transports; `g.staple_description` (the staple of a link in a loop description) |
| `lib/gpt/core/group/differentiable_functional.py` | action functional + `assert_gradient_error` |
| `lib/gpt/core/object_type/su_n.py` | SU(N) group/algebra otypes, generators, conversions |
| `lib/gpt/qcd/gauge/action/wilson.py` | Wilson action (AD stencil value/force; hand-written staples for the heatbath) |
| `lib/gpt/qcd/gauge/smear/differentiable.py` | `dft_diffeomorphism` (Jacobian machinery) |
| `tests/ad/ad.py` | 1st-derivative force checks (first-order tests belong here) |
| `tests/ad/higher_order.py` | 2nd/3rd-order reference of recorded passes (executable spec) |
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
5. Run `bash tests/run` (no args) for the full suite; every test should pass.
6. To compare against an earlier version, extract it with
   `git archive <commit> lib/gpt | tar -x -C <scratch>` and run with
   `PYTHONPATH=<scratch>/lib:$PYTHONPATH` (valid while `lib/cgpt` is
   unchanged).  Do not use `git stash` / `git stash pop` for this: on a clean
   tree the stash saves nothing and the pop applies an older, unrelated
   stash.
