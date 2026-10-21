#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Site-local programs: a straight-line program of complex scalar operations
# (sums, products, weights, conj) on the components of a few fields, run as
# ONE kernel (a tensor stencil, g.stencil.tensor) over all sites.
#
#     prog = g.local_program(function, inputs=[otype, ...], outputs=[otype, ...])
#     y0, y1, ... = prog(x0, x1, ...)
#
# function(*symbolic_inputs) returns one symbolic value per output: a matrix
# input (otype shape (n, n)) is a local_program.matrix of n x n scalars, a
# singlet (shape (1,)) a scalar, a vector (shape (n,)) a list of n scalars.
# The fields are lattices of one precision on one grid (any otypes; real
# fields are stored as complex numbers too).  Symbolic values are
# hash-consed (a subexpression built twice is one node), so the program
# evaluates every distinct subexpression once, and only those the outputs
# depend on.  The kernel keeps the intermediates in per-site
# registers (scratch fields, allocated by liveness).
#
# Derived programs (symbolic, closed under derivation):
#   adjoint(needed)    the vjp: inputs + one flow per output -> the flows of
#                      the inputs needed (conjugate-linear convention of the
#                      reverse AD: flow of a in c = a b is conj(b) flow_c,
#                      of a in c = conj(a) is conj(flow_c))
#   tangent(fiber, k)  the jvp for k directions: inputs + k tangents of the
#                      inputs fiber (direction major) -> k tangents of the
#                      outputs (direction major)
# Both are programs again, so their derivatives are derived the same way
# (with nodes as fields, see ad/reverse/foundation/program.py: a
# self-similar tower to any order).
#
import weakref
import numpy as np
import gpt as g


# ---------------------------------------------------------------------------
# symbolic scalars
# ---------------------------------------------------------------------------
class scalar:
    # an immutable, hash-consed node: op in ("in", "const", "add", "mul",
    # "scale", "conj"); w the input (index, element), the constant or the
    # weight; n a serial number (deterministic code for a deterministic
    # construction order)
    __slots__ = ("op", "args", "w", "n", "__weakref__")

    def __add__(self, other):
        return add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return add(self, scale(-1.0, _scalar(other)))

    def __rsub__(self, other):
        return add(_scalar(other), scale(-1.0, self))

    def __mul__(self, other):
        if isinstance(other, matrix):
            return NotImplemented
        return mul(self, other)

    __rmul__ = __mul__

    def __neg__(self):
        return scale(-1.0, self)

    def __truediv__(self, number):
        return scale(1.0 / complex(number), self)

    def __repr__(self):
        return f"scalar({self.op}, {self.w}, #{self.n})"


_table = weakref.WeakValueDictionary()
_serial = [0]


def _make(op, args, w=None):
    key = (op, tuple(a.n for a in args), w)
    x = _table.get(key)
    if x is None:
        x = scalar()
        x.op, x.args, x.w = op, tuple(args), w
        x.n = _serial[0]
        _serial[0] += 1
        _table[key] = x
    return x


def const(c):
    return _make("const", (), complex(c))


def _scalar(x):
    return x if isinstance(x, scalar) else const(x)


def _is_const(x, c=None):
    return x.op == "const" and (c is None or x.w == c)


def leaf(index, element):
    return _make("in", (), (index, element))


def add(a, b):
    a, b = _scalar(a), _scalar(b)
    if _is_const(a) and _is_const(b):
        return const(a.w + b.w)
    if _is_const(a, 0):
        return b
    if _is_const(b, 0):
        return a
    if b.n < a.n:
        a, b = b, a
    return _make("add", (a, b))


def scale(w, a):
    w, a = complex(w), _scalar(a)
    if w == 1:
        return a
    if w == 0 or _is_const(a, 0):
        return const(0)
    if _is_const(a):
        return const(w * a.w)
    if a.op == "scale":
        return scale(w * a.w, a.args[0])
    return _make("scale", (a,), w)


def mul(a, b):
    a, b = _scalar(a), _scalar(b)
    if _is_const(a):
        return scale(a.w, b)
    if _is_const(b):
        return scale(b.w, a)
    # (weights in front: c (w a) = w (c a), so products of scaled values
    # share their unscaled product)
    if a.op == "scale":
        return scale(a.w, mul(a.args[0], b))
    if b.op == "scale":
        return scale(b.w, mul(a, b.args[0]))
    if b.n < a.n:
        a, b = b, a
    return _make("mul", (a, b))


def conj(a):
    if isinstance(a, matrix):
        return matrix([[conj(x) for x in row] for row in a.rows])
    a = _scalar(a)
    if _is_const(a):
        return const(a.w.conjugate())
    if a.op == "conj":
        return a.args[0]
    if a.op == "scale":
        return scale(a.w.conjugate(), conj(a.args[0]))
    return _make("conj", (a,))


def real(a):
    return scale(0.5, add(a, conj(a)))


def imag(a):
    return scale(-0.5j, add(a, scale(-1.0, conj(a))))


class matrix:
    # an n x n matrix of symbolic scalars
    def __init__(self, rows):
        self.rows = [[_scalar(x) for x in row] for row in rows]
        self.n = len(self.rows)

    @staticmethod
    def identity(n):
        return matrix([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])

    def __getitem__(self, ij):
        return self.rows[ij[0]][ij[1]]

    def __add__(self, other):
        return matrix([[self[i, j] + other[i, j] for j in range(self.n)] for i in range(self.n)])

    def __sub__(self, other):
        return matrix([[self[i, j] - other[i, j] for j in range(self.n)] for i in range(self.n)])

    def __neg__(self):
        return matrix([[-x for x in row] for row in self.rows])

    def __mul__(self, other):
        n = self.n
        if isinstance(other, matrix):
            return matrix(
                [[_sum(self[i, l] * other[l, j] for l in range(n)) for j in range(n)] for i in range(n)]
            )
        return matrix([[x * other for x in row] for row in self.rows])

    def __rmul__(self, other):
        return matrix([[other * x for x in row] for row in self.rows])

    def adj(self):
        return matrix([[conj(self[j, i]) for j in range(self.n)] for i in range(self.n)])

    def trace(self):
        return _sum(self[i, i] for i in range(self.n))


def _sum(terms):
    r = const(0)
    for t in terms:
        r = add(r, t)
    return r


def trace(a):
    return a.trace()


def adj(a):
    return a.adj() if isinstance(a, matrix) else conj(a)


# ---------------------------------------------------------------------------
# values <-> element lists
# ---------------------------------------------------------------------------
def _n_elements(otype):
    return int(np.prod(otype.shape))


def _symbolic(index, otype):
    # the symbolic value of input index of type otype
    shape = otype.shape
    if shape == (1,):
        return leaf(index, 0)
    if len(shape) == 2 and shape[0] == shape[1]:
        n = shape[0]
        return matrix([[leaf(index, i * n + j) for j in range(n)] for i in range(n)])
    if len(shape) == 1:
        return [leaf(index, i) for i in range(shape[0])]
    raise NotImplementedError(f"local_program: inputs of shape {shape}")


def _elements(value, otype):
    # the element list (row major) of a symbolic value of type otype
    shape = otype.shape
    if shape == (1,):
        assert not isinstance(value, (matrix, list)), "local_program: a singlet output is a scalar"
        return [_scalar(value)]
    if isinstance(value, matrix):
        assert len(shape) == 2 and shape == (value.n, value.n)
        return [value[i, j] for i in range(value.n) for j in range(value.n)]
    assert len(shape) == 1 and len(value) == shape[0]
    return [_scalar(x) for x in value]


# ---------------------------------------------------------------------------
# lowering to tensor-stencil code
# ---------------------------------------------------------------------------
# registers per scratch field (an instantiated vector type)
registers_per_field = 30


def _topological(roots):
    # the nodes the roots depend on, children first, and their use counts
    order, seen, uses = [], set(), {}
    for r in roots:
        uses[r.n] = uses.get(r.n, 0) + 1
    stack = [(r, False) for r in reversed(roots)]
    while stack:
        x, expanded = stack.pop()
        if expanded:
            order.append(x)
            continue
        if x.n in seen:
            continue
        seen.add(x.n)
        stack.append((x, True))
        for a in reversed(x.args):
            uses[a.n] = uses.get(a.n, 0) + 1
            if a.n not in seen:
                stack.append((a, False))
    return order, uses


def _lower(roots, field_of_input, n_in_fields):
    # roots: per output field a list of element nodes.  Returns (code,
    # n_scratch_fields).  Fields: the outputs, the inputs, the unit field,
    # then the scratch fields (referenced as temporaries: negative indices)
    ti = g.stencil.tensor_instructions
    flat = [(o, e, x) for o, elems in enumerate(roots) for e, x in enumerate(elems)]
    _, uses = _topological([x for _, _, x in flat])
    is_root = {}
    for o, e, x in flat:
        is_root.setdefault(x.n, (o, e))
    n_out = len(roots)
    one_field = n_out + n_in_fields
    scratch0 = one_field + 1

    def absorbable(x):
        return uses[x.n] == 1 and x.n not in is_root and x.op not in ("in", "const")

    def factor_terms(coef, x, top):
        # x as (coef, [factor nodes], conj_first) terms of a linear
        # combination; nodes not absorbed are read as materialized values
        if top or absorbable(x):
            if x.op == "add":
                return factor_terms(coef, x.args[0], False) + factor_terms(coef, x.args[1], False)
            if x.op == "scale" and x.w == -1:
                return factor_terms(-coef, x.args[0], False)
            if x.op == "mul":
                a, b = x.args
                if absorbable(b) and b.op == "conj" and not (absorbable(a) and a.op == "conj"):
                    a, b = b, a
                if absorbable(a) and a.op == "conj":
                    return [(coef, [a.args[0], b], True)]
                return [(coef, [a, b], False)]
            if x.op == "conj":
                return [(coef, [x.args[0]], True)]
        return [(coef, [x], False)]

    # the plan: one unit per materialized node, children first
    plan = []
    planned = set()
    # (iterative post order over the materialized nodes)
    stack = [(x, False) for _, _, x in reversed(flat)]
    while stack:
        x, expanded = stack.pop()
        if x.n in planned or x.op == "in":
            continue
        if x.op == "const":
            unit = ("const", x, x.w)
            reads = []
        elif x.op == "scale" and x.w != -1:
            unit = ("scale", x, x.w)
            reads = [x.args[0]]
        else:
            terms = factor_terms(1.0, x, True)
            unit = ("terms", x, terms)
            reads = [f for (_, fl, _) in terms for f in fl]
        if not expanded:
            stack.append((x, True))
            for r in reversed(reads):
                if r.n not in planned and r.op != "in":
                    stack.append((r, False))
            continue
        planned.add(x.n)
        plan.append((unit, reads))

    # liveness: the last unit reading each materialized node
    last = {}
    for k, (unit, reads) in enumerate(plan):
        for r in reads:
            last[r.n] = k
    loc = {}
    free, n_regs = [], 0

    def where(x):
        if x.op == "in":
            i, e = x.w
            f, e = field_of_input(i, e)
            return (f, e)
        return loc[x.n]

    def ref(x):
        f, e = where(x)
        return (f, 0, e)

    code = []
    for k, (unit, reads) in enumerate(plan):
        kind, x, data = unit
        if x.n in is_root:
            target = is_root[x.n]
        else:
            if free:
                target = free.pop()
            else:
                target = (-(scratch0 + n_regs // registers_per_field), n_regs % registers_per_field)
                n_regs += 1
        tf, te = target
        if kind == "const":
            code.append((tf, te, ti.mul, data, [(one_field, 0, 0)]))
        elif kind == "scale":
            code.append((tf, te, ti.mul, data, [ref(x.args[0])]))
        else:
            terms = data
            if len(terms) == 2 and all(c == 1 and len(fl) == 1 and not cj for c, fl, cj in terms):
                code.append((tf, te, ti.add, 1.0, [ref(terms[0][1][0]), ref(terms[1][1][0])]))
            else:
                for j, (c, fl, cj) in enumerate(terms):
                    assert c in (1, -1)
                    if j == 0:
                        ins = (ti.mov if c == 1 else ti.mov_neg) if not cj else (ti.mov_cc if c == 1 else ti.mov_neg_cc)
                    else:
                        ins = (ti.inc if c == 1 else ti.dec) if not cj else (ti.inc_cc if c == 1 else ti.dec_cc)
                    code.append((tf, te, ins, 1.0, [ref(f) for f in fl]))
        loc[x.n] = target
        for r in set(r.n for r in reads):
            if last.get(r) == k and r not in is_root and r in loc and loc[r][0] < 0:
                free.append(loc[r])

    # outputs that are inputs, or the same node as an earlier output element
    for o, e, x in flat:
        if x.op == "in" or is_root[x.n] != (o, e):
            code.append((o, e, ti.mov, 1.0, [ref(x)]))
    for c in code:
        assert len(c[4]) <= 2, "local_program: at most two factors per instruction"
    n_scratch = (n_regs + registers_per_field - 1) // registers_per_field
    return code, n_scratch


# ---------------------------------------------------------------------------
# programs
# ---------------------------------------------------------------------------
_ones = {}
_scratch = {}


def _unit_field(grid):
    # a plain complex singlet 1 per grid (kernel workspace, never modified)
    if grid not in _ones:
        one = g.complex(grid)
        one[:] = 1.0
        _ones[grid] = one
    return _ones[grid]


def _scratch_fields(grid, n):
    # kernel-owned register fields per grid (only alive during one kernel)
    fields = _scratch.setdefault(grid, [])
    while len(fields) < n:
        fields.append(g.vcomplex(grid, registers_per_field))
    return fields[0:n]


class local_program:
    scalar = scalar
    matrix = matrix

    def __init__(self, function=None, inputs=None, outputs=None, roots=None):
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        if roots is None:
            symbols = [_symbolic(i, t) for i, t in enumerate(self.inputs)]
            values = function(*symbols)
            if not isinstance(values, (list, tuple)):
                values = [values]
            assert len(values) == len(self.outputs), "local_program: one value per output"
            roots = [_elements(v, t) for v, t in zip(values, self.outputs)]
        self.roots = roots
        self._kernels = {}
        self._derived = {}

    # -- plain evaluation --------------------------------------------------
    def __call__(self, *fields):
        assert len(fields) == len(self.inputs), "local_program: one field per input"
        # (the first field on a derived foundation, e.g. a node, decides; see
        # _foundation in core/transform.py)
        foundation = g.lattice.foundation
        for f in fields:
            fd = getattr(f.__class__, "foundation", None)
            if fd is not None and fd is not g.lattice.foundation:
                foundation = fd
                break
        return foundation.local_program(self, *fields)

    def _kernel(self, grid):
        if grid not in self._kernels:
            n_in = len(self.inputs)
            code, n_scratch = _lower(self.roots, lambda i, e: (len(self.outputs) + i, e), n_in)
            template = g.lattice(grid, self.outputs[0])
            st = g.stencil.tensor(template, [(0,) * grid.nd], code, [(len(code), 1)]) if code else None
            self._kernels[grid] = (st, n_scratch, len(code))
        return self._kernels[grid]

    def plain(self, *fields):
        # the outputs on plain lattices
        fields = [g(f) if isinstance(f, g.expr) else f for f in fields]
        grid = fields[0].grid
        # (one kernel reads all fields with the vector type of the first: one
        # grid and precision; the otypes may differ, each field is addressed
        # by its own number of elements, and every lattice is stored as
        # complex numbers, real ones included)
        for f, t in zip(fields, self.inputs):
            assert len(f.v_obj) == 1 and _n_elements(f.otype) == _n_elements(t), (
                f"local_program: input of type {f.otype.__name__} for {t.__name__}"
            )
            assert f.grid.obj == grid.obj and f.grid.precision is grid.precision, (
                "local_program: all fields on one grid (and precision)"
            )
        st, n_scratch, _ = self._kernel(grid)
        outs = [g.lattice(grid, t) for t in self.outputs]
        if st is not None:
            st(*outs, *fields, _unit_field(grid), *_scratch_fields(grid, n_scratch))
        return outs

    def instructions(self, grid):
        return self._kernel(grid)[2]

    # -- derived programs --------------------------------------------------
    def select(self, outputs):
        # the program of a subset of the outputs
        key = ("select", tuple(outputs))
        if key not in self._derived:
            self._derived[key] = local_program(
                inputs=self.inputs,
                outputs=[self.outputs[o] for o in outputs],
                roots=[self.roots[o] for o in outputs],
            )
        return self._derived[key]

    def adjoint(self, needed):
        # inputs + output flows -> the flows of the inputs needed
        needed = tuple(needed)
        key = ("adjoint", needed)
        if key not in self._derived:
            n_in = len(self.inputs)
            flat = [x for elems in self.roots for x in elems]
            order, _ = _topological(flat)
            flow = {}

            def push(x, f):
                if not _is_const(f, 0):
                    flow[x.n] = add(flow[x.n], f) if x.n in flow else f

            for o, elems in enumerate(self.roots):
                for e, x in enumerate(elems):
                    push(x, leaf(n_in + o, e))
            leaves = {}
            for x in reversed(order):
                f = flow.get(x.n)
                if f is None:
                    continue
                if x.op == "in":
                    leaves[x.w] = f
                elif x.op == "add":
                    push(x.args[0], f)
                    push(x.args[1], f)
                elif x.op == "mul":
                    a, b = x.args
                    push(a, mul(f, conj(b)))
                    push(b, mul(f, conj(a)))
                elif x.op == "scale":
                    push(x.args[0], scale(x.w.conjugate(), f))
                elif x.op == "conj":
                    push(x.args[0], conj(f))
            roots = [
                [leaves.get((i, e), const(0)) for e in range(_n_elements(self.inputs[i]))]
                for i in needed
            ]
            self._derived[key] = local_program(
                inputs=self.inputs + self.outputs,
                outputs=[self.inputs[i] for i in needed],
                roots=roots,
            )
        return self._derived[key]

    def tangent(self, fiber, k):
        # inputs + k tangents of the inputs fiber (direction major) -> the k
        # tangents of the outputs (direction major)
        fiber = tuple(fiber)
        key = ("tangent", fiber, k)
        if key not in self._derived:
            n_in = len(self.inputs)
            flat = [x for elems in self.roots for x in elems]
            order, _ = _topological(flat)
            roots = []
            for j in range(k):
                t = {}
                for x in order:
                    if x.op == "in":
                        i, e = x.w
                        t[x.n] = leaf(n_in + j * len(fiber) + fiber.index(i), e) if i in fiber else const(0)
                    elif x.op == "const":
                        t[x.n] = const(0)
                    elif x.op == "add":
                        t[x.n] = add(t[x.args[0].n], t[x.args[1].n])
                    elif x.op == "mul":
                        a, b = x.args
                        t[x.n] = add(mul(t[a.n], b), mul(a, t[b.n]))
                    elif x.op == "scale":
                        t[x.n] = scale(x.w, t[x.args[0].n])
                    elif x.op == "conj":
                        t[x.n] = conj(t[x.args[0].n])
                roots += [[t[x.n] for x in elems] for elems in self.roots]
            self._derived[key] = local_program(
                inputs=self.inputs + [self.inputs[i] for _ in range(k) for i in fiber],
                outputs=self.outputs * k,
                roots=roots,
            )
        return self._derived[key]


local_program.conj = staticmethod(conj)
local_program.real = staticmethod(real)
local_program.imag = staticmethod(imag)
local_program.trace = staticmethod(trace)
local_program.adj = staticmethod(adj)
local_program.const = staticmethod(const)
