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
  Running under MPI (reset `PYTHONPATH` first, since `~/.bashrc` already put
  the non-MPI build on it):
  ```bash
  cd ~/GPT/gpt
  export PYTHONPATH=; source lib/cgpt/build-mpi/source.sh
  mpirun -np 2 python3 tests/core/core.py --mpi 1.1.1.2
  mpirun -np 4 python3 tests/ad/ad.py --mpi 1.1.2.2
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
- Known pre-existing failure on this box: `tests/qcd/coarsen.py` OOMs
  (15 GB RAM) at its default size; it passes at half size in every
  dimension (`--fgrid 8.8.16.8 --cgrid 4.4.4.4 --ccgrid 2.2.2.4`), which
  still covers the padded 33-point stencil. Everything else should pass.
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
  `wilson` and `improved_with_rectangle` (iwasaki, symanzik, dbw2) take
  value and force from the AD stencil action in
  `qcd/gauge/action/staple_stencil.py` (see §4.7); their hand-written
  `staples()` / `staple()` remain for the heatbath.  `g.qcd.gauge.smear.local_stout`
  is a thin wrapper around `directional_parallel_transport`.
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
  - `with_value=False` (with gradients, without `retain_values`): the root
    value is not needed, so only the values some backward reads are
    computed and `None` is returned (see §4.8).  `functional.gradient` uses
    it; pass it for contractions whose value you do not read, e.g. an HVP
    pass `c(with_value=False)`.
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
  `value_depth_static`, `get_*_container`.
- `ad/reverse/foundation/` — the "foundation" layer: lattice-level
  implementations of ops the node layer dispatches to (trace/sum
  backprops, `where`, `astype`, group conversions), plus
  `foundation/matrix/exp.py` — **`matrix.exp` on lattice nodes is a tower
  of fused kernels**: exp is D_0 of the family D_k(X; H_1..H_k) =
  d^k exp_X(H_1..H_k), whose reverse flows are again D's at X^dag
  (D_{k+1} into X, D_k into H_i), so the gradient of exp is exp.  Each
  plain D_k is one compiled local stencil (multi-dual scaling-squaring +
  Paterson-Stockmeyer Taylor).  All D_k of one tower (the user's exp node
  and the flows built from it, at any depth) share a `_tower`: X is
  identified by identity, and its scaling (the norm bound, which is the same
  for X and X^dag) and the materialized X^dag are computed once.  Non-lattice
  (tensor/scalar) nodes still use the node-op Taylor graph.
- `ad/reverse/foundation/__init__.py` also holds single-node
  **projections**: `traceless_anti_hermitian` / `traceless_hermitian` (the
  `qcd.gauge.project` functions dispatch here for nodes).  They are
  self-adjoint w.r.t. Re tr(a^dag b), so the backward is the same projection
  (one level down for nested flows).  The su(N) group conversions
  (`infinitesimal_to_cartesian`) are written with them.
- (Removed 2026-09-30: *expression nodes*, one node per sum of products
  with generated kernels, `ad/reverse/expression.py`.  In a clean comparison
  against the per-operation graph they won only on the cshift-graph HVP at
  8^4 (-14%), lost at 16^4 (+17%) and were neutral to +9% slower elsewhere;
  absorbing operands at construction also duplicated shared
  subexpressions (2.7x the products).  The last version is in git history
  (commit b7e27c86); do not re-add without a new idea.)
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
  communication phase).  Same products in the same order as the padded
  stencil, so results are bit-identical.  `comm_type=1` (no communication)
  is the plain local kernel (single point, padded fields).
- **Padded stencils** remain only for checkerboarded grids (and with
  `core/stencil/matrix.py`'s switch `use_padded = True`, for comparisons).
  They trust `data_access_hints`: fields are copied into halo-padded fields,
  fields the code does not reference share one scratch field, and declared
  write fields whose first write is fresh are not copied; every field read
  as a factor must be declared as read (referenced but undeclared fields,
  e.g. temporaries, are still copied).  The general stencil ignores the
  hints, but callers keep declaring them for the padded case.
- **Allocator effects in benchmarks**: the padded path frees fields larger
  than a lattice, which raises glibc's dynamic mmap threshold, so later
  lattices of a run are served from the heap without page faults.  Without
  that, a 16^4 color-matrix lattice (9.4 MB) is mmapped and unmapped per
  allocation: local_stout E4 looked 6% slower on the general stencil
  although its stencil kernels got faster.  Compare such runs with
  `MALLOC_MMAP_THRESHOLD_=268435456 MALLOC_TRIM_THRESHOLD_=1073741824`.
- **Peak memory**: large fields and retained graphs live as long as a Python
  name refers to them.  In long drivers (e.g. a loop of reverse passes)
  `del` each pass's graph before building the next, and release large
  temporaries (ng x ng adjoint matrices are 7x a color matrix for SU(3))
  right after their last use.
- **Memory**: nested (2-deep/3-deep) graphs over gauge fields are expensive.
  Resolve results to plain lattices (`value_of` loop) to release graphs as
  soon as you're done reading them; use the smallest grid that exercises the
  code path; avoid holding multiple deep graphs alive at once.
- **Re-running a graph**: each backward pass starts every gradient at
  `None` (an unbuilt zero, see `util.accumulate`), so gradients do not carry
  over between passes; read a leaf's `.gradient` before the next pass.  By
  default the backward frees forward values, so each call re-runs the
  forward.  `root(initial_gradient=..., retain_values=True)` keeps them: repeated reverse
  passes (one per seed) over unchanged leaves share one forward, and
  `root(with_gradients=False, retain_values=True)` returns the root value
  with all intermediates kept, so a seed built from it shares nodes with the
  following reverse pass (as `directional_parallel_transport` does).  Only
  retain values while the leaves are unchanged.
- **Stencil `accumulate` is a field index, not a flag**: with several
  targets in one fused stencil, each target's rewrites must accumulate
  into *its own* field index (target 1 uses `accumulate: 1`, not `0`).
- **Functional calls leave leaf values stale**: `node_differentiable_functional`
  `__call__`/`gradient` overwrite each leaf's `.value` with whatever fields
  they were handed, and `assert_gradient_error` ends on finite-difference
  *composed* lattices. Reusing those leaf nodes in a later section silently
  differentiates the perturbed fields (small, confusing reference
  mismatches). Re-wrap fresh `rad.node(...)` leaves per section.
- **Initial gradients must match the node depth**: a 1-deep node's
  `initial_gradient` must be a *plain* lattice; a constant direction used in
  a *contraction* of a 2-deep root must be a `rad.node(dir,
  with_gradient=False)`.
- **Node graphs are reference cycles**: the backward closures refer back to
  their nodes, so a graph built per call is only released by Python's cyclic
  garbage collector, which counts objects, not bytes -- large fields pile up
  between collections (seen as +75 MB per call at 16^4).  For a repeated
  operation build the graph once and swap the leaf values (as
  `dft_diffeomorphism` and `directional_parallel_transport._local_vjp` do);
  clear root values before re-running (the backward keeps the root's value,
  and `forward` reuses any value that is not None).
- **In-place writes into a gradient** outside `util.accum` must clear
  `node._flow_identity` (as `project` does), or a stale scaled-identity
  record survives (§4.8).
- **Self-accumulating targets are not reads** in `data_access_hints`:
  listing them as read makes the padded stencil (checkerboarded grids) copy
  them in before the kernel overwrites them (the padded wrapper already
  starts a non-fresh target from the caller's value).
- **Flows are adopted, not copied**: the first plain contribution to a
  gradient is adopted as is (`util.accumulate`), so the same field can be
  the gradient of several nodes (both children of an add receive
  `z.gradient`).  Ownership is tracked per gradient slot
  (`node_base._borrowed`): an adopted gradient is copied before an in-place
  update, and leaf gradients are copied if still borrowed when handed out.
  Consequences for new code: a backward closure must return fields it does
  not reuse or overwrite later (no persistent scratch buffers as results),
  and code that writes into a node's gradient in place must call
  `own_gradient()` first (as `project` does).

### 4.7 Stencil nodes (`ad/reverse/foundation/stencil.py`)

A compiled matrix stencil called with node fields (`stencil(out_node,
*input_nodes)`, or `g.parallel_transport_matrix(...)(nodes)`) becomes one
computed node.  Its forward is the compiled kernel; its backward is the
**adjoint code**, derived in closed form as another stencil (the product
rule per factor, shifts negated/relativized, adjoint flags adjusted), so the
gradient of a stencil is a stencil and the tower is self-similar at any
depth.  Three regimes:

- **Temp-free** (factors read inputs only): the adjoint is one kernel; in a
  nested pass the backward is the adjoint stencil as a node again.
- **Accumulation temps** (fields passed as plain constants and used only as
  partial sums): versioned flow slots and a staged adjoint; nested passes
  interpret the adjoint in the node domain (cshift fallback, slow).
- **Local temporaries** (kernel-owned per-site fields, declared with
  `g.stencil.matrix(..., temporaries=[...])`; they are not passed by the
  caller, whose fields and `data_access_hints` are the remaining ones in index
  order).  Rules: (R1) temporaries are written and read at the zero point
  only; (R2) an entry that reads a temporary reads *all* its factors at the
  zero point; temporaries are built from inputs only (no chains), all writes
  of a temporary precede its reads, outputs are never read.  The adjoint
  (`adjoint_code_local`) is two stencils: **stage A** (again with local
  temporaries) recomputes the temporaries and computes the flows of the
  entries that read them (local by R2), writing each temporary's flow
  lambda_T to memory; **stage B** (temp-free) pushes lambda_T, read at shifted
  points, through the temporary definitions into the inputs.  In nested
  passes both stages are node stencils of the same kinds, so higher
  derivatives stay in stencils.  Temporaries need a kernel without
  communication inside (the general stencil exchanges halos before the
  kernel; `comm_type` 1 or 2), never Grid's cartesian stencil.  Use them where a
  subproduct is **shared** (the up/down staples of plaquettes and
  rectangles, `staple_stencil_code`); without sharing they only add a stage
  barrier and the lambda_T memory traffic (the Wilson action is faster as a
  plain plaquette-loop stencil).

Plain-run optimizations:

- **Seedless adjoints**: if the flow into a single-output stencil is exactly
  `c * identity` (see §4.8), the adjoint kernels are compiled once per c
  with the flow factor dropped and c (conj(c) for an adjointed read) folded
  into the weights (`seedless_code`).  For a traced loop sum this removes one
  of k matrix products per k-link entry (72 -> 48 for the Wilson force).
- **Shared padding** (padded stencils only, i.e. checkerboarded grids): the
  forward's halo-padded input copies are handed to the adjoint kernels (same
  padding domain) instead of being copied again;
  use-once, tied to the forward value, checked by object identity
  (switch: `share_padded`).
- **Common-subexpression elimination of the executed kernels**:
  `g.stencil.matrix(..., cse=...)` / `g.local_stencil.matrix(..., cse=...)`
  compiles an execution plan in which repeated adjacent factor pairs (a pair
  and its adjoint reversal share one) become per-site temporaries, possibly
  built from temporaries (`core/local_stencil/cse.py`, greedy Re-Pair).  The
  stencil object keeps its original `points`/`code`/`temporaries` (the
  executed plan is `executed`), so the AD derivation -- including the nested
  tower's `matrix(compiled[0], ...)` -- still derives from uncombined codes,
  while every plain run at any level uses the plan.  Fields the kernel writes
  are combined only at the zero point and per write version (live reads).
  A plan is used only if it saves >= 15% of the products
  (`min_saving_default`): the kernels are dominated by factor fetches, so a
  temporary costs about what it saves (measured: >= 18% fewer products ->
  4-14% faster kernels, <= 12% -> neutral or slower).  Enabled for the AD
  adjoint kernels (switch: `cse` in `foundation/stencil.py`); cartesian-only
  kernels run unpadded without temporaries and are never combined.  Effect:
  fused-loop iwasaki force -13%, iwasaki HVP -3..-4%, 3rd derivative
  -2..-5%; the production forces and local_stout are below the threshold and
  bit-identical.

### 4.8 Value needs and structured flows

- **Which values a backward reads** is declared per node:
  `_reads_children` (None = all children, else per child i the child
  indices the flow into child i reads) and `_reads_self` (default True).
  `node_op(..., reads=...)` sets both (a node_op never reads its own value).
  Declared: products, sums, adj, trace, sum, list-element access, `project`,
  and stencil nodes (the inputs, never the output).  `needed_values` walks the
  graph from the root and `forward` skips every computed node nothing
  needs (only with `with_value=False`).  **Safety net**: `value_of`
  evaluates a missing value on demand, so an undeclared read is still
  correct -- but it cascades (the recomputation needs its own inputs) and
  silently loses the saving.  So a backward must not call `value_of` on
  inputs it does not really need: trace/sum build their identity from the
  container (`_reduction_identity`).
- **Scaled-identity flows**: trace and sum record when the flow they pass
  down is exactly `c * identity` (a scalar flow broadcast back to a field):
  `node._flow_identity = (gradient_object, c)`, read with
  `util.identity_flow_scale(node)`.  It is valid only while `node.gradient`
  is that very object: `util.accum` clears it on every contribution and
  `project` clears it after its in-place update.  Consumers may exploit it
  (the stencil seedless adjoints); all others see an ordinary field.

## 5. File map (AD-relevant)

| Path | Role |
|---|---|
| `lib/gpt/ad/reverse/node.py` | node, node_op, forward/backward, functional |
| `lib/gpt/ad/reverse/util.py` | nodify, product, value_of, `resolve` (plain value of a finished pass's result), containers |
| `lib/gpt/ad/reverse/transform.py` | sin/cos/... node transforms |
| `lib/gpt/ad/reverse/functional_node.py` | a `differentiable_functional` as a node (first order; used by `g.ml` losses) |
| `lib/gpt/ad/reverse/preimage.py` | the preimage x = phi^-1(y) of a diffeomorphism as nodes (first order; backward: solve J_xx^T lambda = c with `dfm.jacobian`, flows lambda and -(dphi/d others)^T lambda); `directional_parallel_transport.inv` accepts nodes through it |
| `lib/gpt/ad/reverse/foundation/` | lattice-level op backprops; projection nodes; `matrix/exp.py` (exp tower) |
| `lib/gpt/ad/reverse/foundation/stencil.py` | node foundation for compiled matrix stencils (§4.7): adjoint derivation (`adjoint_code`, `adjoint_code_local`), multi-output list nodes, local temporaries, seedless adjoints, shared padding |
| `lib/gpt/core/stencil/matrix.py`, `lib/gpt/core/local_stencil/matrix.py` | compiled matrix stencils (kind selection, `comm_type`); padded wrapper (checkerboarded grids); `temporaries=`; `cse=` |
| `lib/cgpt/lib/foundation/general_stencil.h` | general stencil: geometry (lookup table, halo transfer plan), per-field halo, batched exchange via Grid's `StencilSendToRecvFrom`, manager (field point sets); used by `stencil/matrix.h` (`comm_type == 2`, kernel loops in `stencil/matrix_loops.h`) |
| `lib/gpt/core/local_stencil/cse.py` | common-subexpression elimination of a kernel's execution plan (tested in `tests/core/stencil.py`) |
| `tests/ad/stencil.py` | stencil AD: fused two-output stencil, path-based stencils, local temporaries (staple action vs cshift graph) at 1st/2nd/3rd order, `with_value=False` |
| `lib/gpt/qcd/gauge/action/staple_stencil.py` | gauge action value/force as AD stencils (plaquette loop, or staples as local temporaries with rectangles) |
| `lib/gpt/qcd/gauge/smear/directional_parallel_transport.py` | checkerboarded smearing (behind `local_stout`): local jacobian/VJP via the staple (`g.staple_description`), the per-site Jacobian block (`jacobian_matrix`, optionally at a prescribed staple), log-det and its force (also weighted per site), `inv` (`g.algorithms.nonlinear.fixed_point` with site-local Newton steps; `inverse_history`) |
| `lib/gpt/ad/forward/` | series / Landau differential algebra |
| `lib/gpt/core/group/operation.py` | inner_product/compose dispatch, `zero` (cartesian zero) |
| `lib/gpt/algorithms/nonlinear/fixed_point.py` | `g.algorithms.nonlinear.fixed_point`: iteration control of x = Phi(x) (history, contraction rate, switch to an accelerated step, failure), `fixed_point.newton(residual, solve)` Newton steps; used by `directional_parallel_transport.inv` |
| `lib/gpt/core/group/algebra_kernels.py` | `g.group.algebra_kernels`: site-local kernels between SU(N) algebra fields and adjoint matrices (coordinates as rows, combinations of generators) |
| `lib/gpt/core/parallel_transport/matrix.py` | weighted parallel transports; `g.staple_description` (the staple of a link in a loop description) |
| `lib/gpt/core/group/differentiable_functional.py` | action functional + `assert_gradient_error` |
| `lib/gpt/core/object_type/su_n.py` | SU(N) group/algebra otypes, generators, conversions |
| `lib/gpt/qcd/gauge/action/wilson.py` | Wilson action (AD stencil value/force; hand-written staples for the heatbath) |
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
6. To compare against an earlier version, extract it with
   `git archive <commit> lib/gpt | tar -x -C <scratch>` and run with
   `PYTHONPATH=<scratch>/lib:$PYTHONPATH` (valid while `lib/cgpt` is
   unchanged).  Do not use `git stash` / `git stash pop` for this: on a clean
   tree the stash saves nothing and the pop applies an older, unrelated
   stash.
