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
# Node foundation for compiled matrix stencils: a list of input fields mapped
# to a (list of) output field(s), both as nodes.
#
#     stencil(output, *inputs)
#
# `output` is field(s) 0..m-1: a single node (m=1) or a LIST node (m outputs,
# e.g. a fused kernel computing a plaquette and its adjoint at once).  Each
# input argument is a single node, a LIST node (expanding to one input field
# per element, e.g. the 4 gauge links as one node), or a plain lattice (a
# constant).  The inputs occupy fields m..n-1.
#
# The output node is converted in place into a computed node whose
#   - forward is a SINGLE kernel pass computing all m outputs, and
#   - backward runs the ADJOINT of the stencil code, which is a stencil in
#     closed form (adjoint_code): the same entries with the product rule
#     applied to each factor (shifts negated/relativized, adjoint flags
#     adjusted, the m output flows as extra inputs), written into flow slots.
#     So, as for g.cshift -- whose gradient is again a cshift with the
#     displacement negated -- the gradient of a stencil is a stencil.
#
# The adjoint reads only the (never-written) forward values and output flows,
# never a flow slot, so all of its entries fuse to a single kernel pass.
#
# In a nested (multi-deep) pass the flows are lazy node graphs and cannot be
# fed to the kernels.  The backward is then the ADJOINT STENCIL acting on
# nodes again: it is built as a first-class stencil node whose forward runs
# the compiled adjoint kernel (on plain-resolved operands) and whose backward
# is the adjoint-of-adjoint stencil node.  So the recursion is a self-similar
# tower of stencils -- S, A = adjoint(S), A' = adjoint(A), ... -- with NO
# cshift: the shifts live in the compiled kernel's points, and the flow is a
# child of the adjoint node (its inputs are the output flows, the output
# values, and the inputs).
#
# Because the m outputs are one list node (not m siblings), there is no
# fused multi-output bookkeeping: one node, one flow list, one adjoint run.
#
# Supported regime (the standard GPT stencil pattern: staple,
# parallel_transport_matrix, ...): only outputs are written; factors only
# reference input fields; the first write of each output is fresh
# (accumulate=-1, or adds an input), so an output's old value is not part of
# the computation; rewrites accumulate into the running value.  Per-site
# temporaries are kernel-owned local temporaries (adjoint_code_local).
#
import weakref
import gpt as g
from gpt.ad.reverse.util import value_of, is_node, accum, identity_flow_scale, constant


# share the padded inputs of a stencil node's forward run with its adjoint
# run (see _keep_padded in matrix); a switch for tests and comparisons
share_padded = True

# combine common subexpressions in the executed adjoint kernels (the minimal
# number of uses of a per-site temporary, see core/local_stencil/cse.py;
# False: off).  Only the execution plan changes: the adjoint codes the tower
# derives from are the uncombined ones.
cse = 2


def _padding_domain(K):
    # the halo-padding domain a compiled stencil runs on (None: not padded)
    padding = getattr(K, "padding", None)
    return None if padding is None else padding.domain


def _product_rule(weight, fl, phi, fmap):
    # the flows of one entry z = weight * A_0 * ... * A_{k-1} (factors fl,
    # A_m possibly shifted and adjointed) for the flow phi of its target:
    # per factor m, (field of A_m, weight', factor list) with
    #   flow of A_m = weight' * (the other A_l, in reverse order, each shifted
    #     by p_l - p_m relative to factor m) * (phi shifted by -p_m),
    # weight' = conj(weight) and the adjoint flag of every other factor
    # flipped if A_m was not adjointed, both unchanged if it was.  (The
    # adjoint of a shifted/adjointed matrix is again a shifted/adjointed
    # matrix -- the same rule as the gradient of g.cshift.)  fmap maps the
    # factors' field indices to the adjoint code's layout.
    k = len(fl)
    res = []
    for m in range(k):
        i_m, p_m, a_m = fl[m]

        def rel(l, p_m=p_m):
            return tuple(x - y for x, y in zip(fl[l][1], p_m))

        phi_ref = (phi, tuple(-x for x in p_m), a_m)
        if a_m == 0:
            f = [(fmap(fl[l][0]), rel(l), 1 - fl[l][2]) for l in range(m - 1, -1, -1)]
            f += [phi_ref]
            f += [(fmap(fl[l][0]), rel(l), 1 - fl[l][2]) for l in range(k - 1, m, -1)]
            w = complex(weight).conjugate()
        else:
            f = [(fmap(fl[l][0]), rel(l), fl[l][2]) for l in range(m + 1, k)]
            f += [phi_ref]
            f += [(fmap(fl[l][0]), rel(l), fl[l][2]) for l in range(m)]
            w = weight
        res.append((i_m, w, f))
    return res


def adjoint_code(code, outputs, flowed, ndim):
    # The gradient of a stencil, in closed form, as another stencil.
    #
    #   code    : forward entries (target, accumulate, weight, factors),
    #             factors = [(field, point_tuple, adj_flag), ...]; only the
    #             outputs are written
    #   outputs : field indices that are stencil outputs (their flows are
    #             supplied by the caller)
    #   flowed  : the input field indices whose flows are computed (the other
    #             inputs' entries are dropped)
    #   ndim    : point dimensionality
    #
    # Entry by entry, with psi the supplied flow of the entry's target: the
    # flows of the factors by the product rule (_product_rule), and the flow
    # of an accumulate read of an input is psi.
    #
    # Field layout of the adjoint code:
    #
    #     [slot_i for i in flowed] + [psi_t for t in outputs]
    #     + [value_i for i in range(n_fields)]
    #
    #   slots  : one flow slot per flowed input, the targets of the adjoint
    #            code; the first write of each is fresh
    #   psi    : the supplied flow of each output (pure inputs)
    #   values : the forward values (pure inputs)
    #
    # The adjoint reads no slot, so it is a valid forward stencil of a single
    # kernel pass.  Every flowed input is referenced (see matrix), so every
    # slot is written.
    zero = (0,) * ndim
    outs = sorted(set(outputs))
    slot = {i: r for r, i in enumerate(flowed)}
    n_comp = len(flowed)
    PSI = lambda t: n_comp + outs.index(t)
    F = lambda i: n_comp + len(outs) + i

    entries = []
    first_write = set()

    def emit(target, weight, flist):
        acc = -1 if target not in first_write else target
        first_write.add(target)
        entries.append((target, acc, weight, flist))

    for target, acc, weight, factors in reversed(code):
        psi = PSI(target)
        for i_m, w, flist in _product_rule(weight, factors, psi, F):
            if i_m in slot:
                emit(slot[i_m], w, flist)
        if acc != -1 and acc != target and acc in slot:
            emit(slot[acc], 1.0, [(psi, zero, 0)])
    return entries


def seedless_code(code, psi_index, c):
    # the code for a flow psi = c * identity (see util.identity_flow_scale):
    # the psi factor of each entry is dropped and c (conj(c) for an adjointed
    # read) folded into the weight -- psi is constant, so its point does not
    # matter.  An entry whose only factor is psi keeps it (psi is still a
    # valid field).
    out = []
    for (tt, ac, w, fl) in code:
        k = [i for i, (f, p, a) in enumerate(fl) if f == psi_index]
        if len(k) == 1 and len(fl) > 1:
            f, p, a = fl[k[0]]
            w = w * (c.conjugate() if a else c)
            fl = fl[: k[0]] + fl[k[0] + 1 :]
        out.append((tt, ac, w, fl))
    return out


# compiled seedless kernels are cached per (stencil adjoint, c); the scale is
# normally a fixed number (e.g. -1/2 for Re tr), a few variants are kept
seedless_cache_size = 8


def _seedless_kernels(cache, key, c, build):
    variants = cache.setdefault(("seedless",) + key, {})
    if c not in variants:
        if len(variants) >= seedless_cache_size:
            variants.pop(next(iter(variants)))
        variants[c] = build(c)
    return variants[c]


def adjoint_code_local(code, outputs, temps, inputs, ndim):
    # The gradient of a stencil with LOCAL temporaries (kernel-owned per-site
    # fields, see g.local_stencil.matrix) as two stencils.  Regime:
    #   R1: temporaries are read and written at the zero point only,
    #   R2: an entry that reads a temporary reads all its factors at the zero
    #       point,
    # temporaries are built from inputs only (no chains), all writes of a
    # temporary precede its reads, outputs are never read, and every write is
    # fresh or accumulates into its own target.  Then (psi = output flows,
    # lam_T = flow of temporary T):
    #   stage A (a stencil with local temporaries): recomputes the
    #     temporaries and runs the reverse sweep over the entries writing
    #     outputs; by R2 the flows of entries reading temporaries are local,
    #     so they go into the input slots and into lam_T (written to memory)
    #   stage B (temp-free): the flows of the temporary-defining entries,
    #     which read lam_T at shifted points (a barrier after stage A)
    # Both stages are stencils of the supported kinds again, so the adjoint of
    # the adjoint is derived the same way (a self-similar tower).
    #
    # Field layouts (nI inputs, nT temporaries, m outputs, in index order):
    #   stage A: [slot_i]*nI [lam_T]*nT [psi_o]*m [value_i]*nI [local T]*nT
    #            (the local T are the kernel's temporaries, not passed)
    #   stage B: [slot_i]*nI [lam_T]*nT [value_i]*nI
    # returns (A, locA, B_fresh, B_acc, written_A, written_B); B_fresh starts
    # its slots fresh (a separate contribution), B_acc accumulates into the
    # slots stage A wrote (for the compiled run).
    zero = (0,) * ndim
    outs, tmps, inp = list(outputs), list(temps), list(inputs)
    first_read = {}
    for j, (t, acc, w, fl) in enumerate(code):
        assert acc in (-1, t), (
            "stencil node mode with local temporaries: writes must be fresh or "
            "accumulate into their own target")
        reads_tmp = any(f in tmps for (f, p, a) in fl)
        for (f, p, a) in fl:
            assert f not in outs, "stencil node mode: outputs must not be read"
            if f in tmps:
                assert p == zero, "local temporaries are read at the zero point (R1)"
                first_read.setdefault(f, j)
            if reads_tmp:
                assert p == zero, (
                    "an entry reading a local temporary reads all its factors at "
                    "the zero point (R2)")
        if t in tmps:
            assert not reads_tmp, "local temporaries built from temporaries are not supported"
    for j, (t, acc, w, fl) in enumerate(code):
        if t in first_read:
            assert j < first_read[t], "all writes of a local temporary precede its reads"
    for t in outs:
        firsts = [acc for (tt, acc, w, fl) in code if tt == t]
        assert not firsts or firsts[0] == -1, (
            "stencil node mode: first write of output %d must be fresh" % t)

    nI, nT, m = len(inp), len(tmps), len(outs)
    SLOT = lambda i: inp.index(i)
    LAM = lambda t: nI + tmps.index(t)
    PSI = lambda o: nI + nT + outs.index(o)
    VAL_A = lambda i: nI + nT + m + inp.index(i)
    LOC = lambda t: nI + nT + m + nI + tmps.index(t)
    FA = lambda f: VAL_A(f) if f in inp else LOC(f)
    VAL_B = lambda i: nI + nT + inp.index(i)

    A, B_fresh, B_acc = [], [], []
    written_A, written_B = set(), set()
    # stage A: recompute the temporaries (kernel-local)
    for (t, acc, w, fl) in code:
        if t in tmps:
            A.append((LOC(t), -1 if acc == -1 else LOC(t), w, [(FA(f), p, a) for (f, p, a) in fl]))
    for (t, acc, w, fl) in reversed(code):
        if t in outs:
            for (i_m, wf, f) in _product_rule(w, fl, PSI(t), FA):
                target = LAM(i_m) if i_m in tmps else SLOT(i_m)
                A.append((target, target if target in written_A else -1, wf, f))
                written_A.add(target)
        else:
            for (i_m, wf, f) in _product_rule(w, fl, LAM(t), VAL_B):
                target = SLOT(i_m)
                B_fresh.append((target, target if target in written_B else -1, wf, f))
                B_acc.append((target, target, wf, f))
                written_B.add(target)
    locA = [LOC(t) for t in tmps]
    return A, locA, B_fresh, B_acc, written_A, written_B


def _compile(grid, otype, code, temporaries=()):
    # a compiled matrix stencil for code with point tuples; the data access
    # hints are in the positions of the passed fields (without temporaries)
    ndim = grid.nd
    temps = sorted(temporaries)
    pos = lambda i: i - sum(1 for t in temps if t < i)
    pts = sorted({(0,) * ndim} | {p for (tt, ac, w, fl) in code for (f, p, a) in fl})
    pm = {p: i for i, p in enumerate(pts)}
    ccode = [(tt, ac, w, [(f, pm[p], a) for (f, p, a) in fl]) for (tt, ac, w, fl) in code]
    K = g.stencil.matrix(g.lattice(grid, otype), pts, ccode, temporaries=temps, cse=cse)
    written = sorted({pos(tt) for (tt, ac, w, fl) in code if tt not in temps})
    # (a target accumulating into itself is not a read: the padded stencil
    # starts a target from the caller's value unless its first write is fresh)
    read = sorted(
        {pos(f) for (tt, ac, w, fl) in code for (f, p, a) in fl if f not in temps}
        | {pos(ac) for (tt, ac, w, fl) in code if ac not in (-1, tt) and ac not in temps}
    )
    K.data_access_hints(written, read, [])
    return K


class _node_setup:
    # what both node modes (matrix, _matrix_local_temporaries) share: the
    # code with point tuples, the output field(s) 0..m-1 (a single node, or a
    # list node, which keeps its list semantics with a single output, e.g.
    # the adjoint of a stencil with one input field) and the inputs (a list
    # node expands to one child per element, a plain value to a constant
    # node)
    def __init__(self, stencil, inner, fields):
        self.stencil = stencil
        points = inner.points
        self.raw = [
            (e["target"], e["accumulate"], e["weight"],
             [(f, points[p], a) for (f, p, a) in e["factor"]])
            for e in inner.code
        ]
        output = fields[0]
        assert is_node(output), "stencil node mode: the output must be a node"
        self.output = output
        self.listed = output._container.tag[0] is list
        self.m = len(output) if self.listed else 1
        self.grid = output.grid
        self.otype = output.otype
        self.children = []
        for arg in fields[1:]:
            if is_node(arg) and arg._container.tag[0] is list:
                self.children.extend(arg[i] for i in range(len(arg)))
            else:
                self.children.append(constant(arg))

    def cache(self, name):
        # the adjoint codes and kernels, cached on the stencil object
        cache = getattr(self.stencil, name, None)
        if cache is None:
            cache = {}
            setattr(self.stencil, name, cache)
        return cache

    def nested(self):
        # node depth: gradient-carrying children must be uniform (constants
        # are plain at any depth and carry no inner dependency).
        # This is decided in the backward pass, not at construction:
        # `value_of` evaluates a computed child, and a value cached before the
        # graph is first run would then be reused by node.forward (which only
        # recomputes values that are None) instead of being rebuilt from the
        # updated leaves.
        node_vals = [is_node(value_of(c)) for c in self.children if c.with_gradient]
        assert all(node_vals) or not any(node_vals), (
            "stencil node mode: gradient-carrying children must have uniform node depth")
        return any(node_vals)

    def psi(self):
        # the m output flows; a single output node has one flow, a list node
        # a list of m (outputs that received no flow are zero)
        self.output.materialize_gradient()
        return self.output.gradient if self.listed else [self.output.gradient]

    def lattices(self, n):
        return [g.lattice(self.grid, self.otype) for _ in range(n)]

    def forward(self, run):
        # forward: one kernel pass computes all m outputs, run(outs, ops) on
        # plain operands.  The operands are resolved exactly ONE level down
        # (value_of, not all the way to plain): a lazy expr is materialized
        # because the kernel needs lattices.
        ops = []
        for c in self.children:
            v = value_of(c)
            if isinstance(v, g.expr):
                v = g(v)
            ops.append(v)

        if any(is_node(v) for v in ops):
            # nested pass: the VALUE of this node must itself be a stencil
            # node one level down, not a plain field.  Consumers backpropagate
            # with value_of(this node), so a plain value would hand them plain
            # operands and the flow psi that reaches this node would carry no
            # dependence on the inputs -- a nonlinear consumer then loses the
            # dpsi/dU half of the 2nd derivative.  So the forward recurses:
            # the same self-similar tower as the backward, bottoming out at
            # plain operands, where the compiled kernel runs.
            inner = self.lattices(self.m)
            inner = g.ad.reverse.node(inner if self.listed else inner[0])
            return matrix(self.stencil, inner, *ops)

        outs = self.lattices(self.m)
        run(outs, ops)
        return outs if self.listed else outs[0]

    def install(self, forward, backward, tag):
        # the output node becomes the computed node
        output = self.output
        output._forward = forward
        output._children = self.children
        output._backward = backward
        output.value = None
        output.gradient = None
        output._tag = tag
        # the backward reads the inputs, never the output value (the nested
        # and padding-sharing paths accept a missing value)
        output._reads_children = None
        output._reads_self = False
        return output


def _matrix_local_temporaries(stencil, inner, fields):
    # node mode for a stencil with kernel-owned local temporaries (see
    # adjoint_code_local): the forward is the compiled kernel as always, the
    # backward is stage A (with local temporaries) followed by stage B
    ns = _node_setup(stencil, inner, fields)
    raw, output, m, listed, children = ns.raw, ns.output, ns.m, ns.listed, ns.children
    grid, otype_t = ns.grid, ns.otype
    temps = list(inner.temporaries)

    # code index of each passed field (the temporaries are not passed)
    n_passed = m + len(children)
    full = [i for i in range(n_passed + len(temps)) if i not in temps]
    assert len(full) == n_passed
    outputs = full[:m]
    inputs = full[m:]
    for (tt, ac, w, fl) in raw:
        assert tt in outputs or tt in temps, "stencil node mode: inputs must not be written"
    referenced = {f for (tt, ac, w, fl) in raw for (f, p, a) in fl if f in inputs}

    cache = ns.cache("_node_adj_local")
    key = (m, n_passed)
    if key not in cache:
        A, locA, B_fresh, B_acc, written_A, written_B = adjoint_code_local(
            raw, outputs, temps, inputs, grid.nd)
        KA = _compile(grid, otype_t, A, locA)
        KB_fresh = _compile(grid, otype_t, B_fresh) if B_fresh else None
        KB_acc = _compile(grid, otype_t, B_acc) if B_acc else None
        nI, nT = len(inputs), len(temps)
        # A's outputs that it never writes but stage B_acc or the caller reads
        unwritten_A = [r for r in range(nI + nT) if r not in written_A]
        cache[key] = (KA, KB_fresh, KB_acc, written_A, written_B, unwritten_A, A, locA)
    KA, KB_fresh, KB_acc, written_A, written_B, unwritten_A, A_code, locA = cache[key]
    nI, nT = len(inputs), len(temps)

    def _KA_for(c):
        # stage A for a flow c * identity (m = 1, see seedless_code); psi is
        # the field after the slots and the temporary flows
        return _seedless_kernels(
            cache, key, c, lambda c: _compile(grid, otype_t, seedless_code(A_code, nI + nT, c), locA))

    def run_fwd():
        return ns.forward(lambda outs, ops: stencil(*(outs + ops)))

    def _backward(z):
        vals = [value_of(c) for c in children]
        psi = list(ns.psi())
        if not ns.nested():
            slots = ns.lattices(nI + nT)
            for r in unwritten_A:
                slots[r][:] = 0
            K = KA
            if not listed:
                c = identity_flow_scale(output)
                if c is not None:
                    K = _KA_for(c)
            K(*(slots + psi + vals))
            if KB_acc is not None:
                KB_acc(*(slots + vals))
            for k, c in enumerate(children):
                if c.with_gradient and inputs[k] in referenced:
                    accum(c, slots[k], 1)
            return
        # nested: both stages as stencil nodes one level down; stage A has
        # local temporaries (this function again), stage B is temp-free
        A_out = g.ad.reverse.node(ns.lattices(nI + nT))
        Anode = matrix(KA, A_out, *psi, *vals)
        Bnode = None
        if KB_fresh is not None:
            B_out = g.ad.reverse.node(ns.lattices(nI))
            Bnode = matrix(KB_fresh, B_out, *[Anode[nI + t] for t in range(nT)], *vals)
        for k, c in enumerate(children):
            if not (c.with_gradient and inputs[k] in referenced):
                continue
            flow = Anode[k] if k in written_A else None
            if Bnode is not None and k in written_B:
                flow = Bnode[k] if flow is None else flow + Bnode[k]
            if flow is not None:
                accum(c, flow, 1)

    return ns.install(
        run_fwd,
        _backward,
        f"stencil({m} output, {len(temps)} local temporaries, {len(inner.code)} lines of code)",
    )


def matrix(stencil, *fields):
    inner = getattr(stencil, "local_stencil", stencil)
    if getattr(inner, "temporaries", ()):
        return _matrix_local_temporaries(stencil, inner, fields)
    ns = _node_setup(stencil, inner, fields)
    raw, output, m, listed, children = ns.raw, ns.output, ns.m, ns.listed, ns.children
    grid, otype_t = ns.grid, ns.otype
    ndim = grid.nd

    # the inputs occupy fields m..n_fields-1; the field count comes from the
    # ARGUMENTS, not from the highest index the code happens to reference: a
    # caller may legitimately pass fields the code never reads
    # (g.parallel_transport hands over all links, whatever directions the
    # paths use), exactly as the compiled kernel allows
    n_fields = m + len(children)
    outputs = list(range(m))

    # regime: only outputs are written, factors reference only inputs, the
    # first write of each output does not read its own old value.  An output
    # the code never writes is valid (it stays at its zero-initialized value)
    # -- this happens for derived adjoint stencils, whose inputs include
    # forward values the adjoint code never references.
    first = {}
    referenced = set()
    for (tt, ac, w, fl) in raw:
        assert tt < m, (
            "stencil node mode: only outputs are written (field %d is not an "
            "output; per-site temporaries are local temporaries)" % tt)
        first.setdefault(tt, ac)
        for (f, p, a) in fl:
            assert m <= f < n_fields, (
                "stencil node mode: factors must reference input fields 0..%d "
                "(field %d)" % (n_fields - 1, f))
            referenced.add(f)
        if ac != -1 and ac != tt:
            assert m <= ac < n_fields, (
                "stencil node mode: accumulate reads an input field (field %d)" % ac)
            referenced.add(ac)
    for t, ac in first.items():
        assert ac != t, (
            "stencil node mode: first write of output %d must not read its "
            "own old value (acc=-1, or acc=input)" % t)

    # the inputs whose flow is needed (referenced, gradient-carrying): the
    # adjoint computes only their slots, so constant inputs cost nothing in
    # the backward at any level
    flowed = tuple(m + k for k, c in enumerate(children) if c.with_gradient and m + k in referenced)
    slot_of = {ci: r for r, ci in enumerate(flowed)}
    n_comp = len(flowed)
    n_off = n_comp + m

    # the adjoint code, in closed form (another stencil), compiled; cached on
    # the stencil object, keyed by the output count and the flowed inputs
    cache = ns.cache("_node_adj")
    key = (m, flowed)
    if key not in cache:
        code = adjoint_code(raw, outputs, flowed, ndim)
        cache[key] = (code, _compile(grid, otype_t, code) if code else None)
    adj_code, compiled = cache[key]

    def _compiled_for(c):
        # the adjoint kernel for a flow c * identity (m = 1, see seedless_code)
        return _seedless_kernels(
            cache, key, c, lambda c: _compile(grid, otype_t, seedless_code(adj_code, n_comp, c)))

    # the adjoint never reads an output's forward value (the regime asserts
    # factors reference inputs and first writes are fresh), but the kernel
    # padding plan needs a valid lattice at every read slot
    dummy = g.lattice(grid, otype_t)

    # Padded inputs shared between the forward run and the adjoint run.
    # A padded (multi-direction) stencil copies each input into a halo-padded
    # field; the adjoint reads the same forward values, so when both kernels
    # run on the same padding domain, the forward's padded inputs are handed
    # to the adjoint kernel instead of being copied again.  Invalidation:
    #   - the padded copies belong to ONE forward value and are used by at
    #     most ONE adjoint run: they are dropped after the first backward of
    #     this node, when that value object dies (weakref callback, e.g. the
    #     graph releases its forward values), or when the next forward run
    #     replaces them.  (Use-once: the root's value is never freed by the
    #     backward and a graph is a reference cycle, so copies tied only to
    #     the value would live until the cyclic garbage collection; further
    #     reverse passes over retained values pad again, as without sharing.)
    #   - they are used only while this node's value is still that object and
    #     every shared input's current value is the very object that was
    #     padded (so a re-run forward, or leaf values replaced between calls,
    #     are never served stale); as for retained forward values in general,
    #     the fields must not be modified in place while the values live.
    # Only the plain (compiled) path shares; nested passes build new nodes.
    # Memory: up to one padded copy per read input is held from the forward
    # to the node's backward (the adjoint allocated the same copies anyway,
    # but only during its run).
    fwd_domain = _padding_domain(stencil)
    shared = {}

    def _forget(token):
        if shared.get("token") is token:
            shared.clear()

    def _keep_padded(outs, raw, padded_fields):
        shared.clear()
        if padded_fields is None:
            return
        read_only = set(stencil.read_fields) - set(stencil.write_fields)
        pads, refs = {}, {}
        for k, v in enumerate(raw):
            i = m + k
            if i not in read_only or i not in referenced:
                continue
            try:
                refs[i] = weakref.ref(v)
            except TypeError:
                continue
            pads[i] = padded_fields[i]
        if not pads:
            return
        token = object()
        shared["token"] = token
        shared["out"] = weakref.ref(outs[0], lambda _r, t=token: _forget(t))
        shared["pads"] = pads
        shared["refs"] = refs

    def _shared_pads(z):
        # the forward's padded inputs if they are still valid for z (see above)
        if not shared:
            return {}
        out = shared["out"]()
        v = z.value
        if out is None or v is None:
            return {}
        if listed:
            if not isinstance(v, list) or len(v) != m:
                return {}
            v = v[0]
        if v is not out:
            return {}
        pads = {}
        for i, P in shared["pads"].items():
            if shared["refs"][i]() is value_of(children[i - m]):
                pads[i] = P
        return pads

    def run_plain(outs, ops):
        if share_padded and fwd_domain is not None and _padding_domain(compiled) is fwd_domain:
            # keep the padded inputs for the adjoint run, see _keep_padded
            raw = [value_of(c) for c in children]
            _keep_padded(outs, raw, stencil(*(outs + ops)))
        else:
            stencil(*(outs + ops))

    def run_fwd():
        return ns.forward(run_plain)

    def run_adj_plain(z):
        # backward, plain flows: the compiled adjoint kernel, one pass, on
        # fresh slot lattices (the first write of every slot is fresh)
        slots = ns.lattices(n_comp)
        full = [dummy] * m  # output forward values are not read
        for c in children:
            v = value_of(c)
            if is_node(v):
                raise NotImplementedError(
                    "stencil node mode: nested (non-plain) values are not supported yet")
            full.append(v)
        for fl in ns.psi():
            if is_node(fl):
                raise NotImplementedError(
                    "stencil node mode: nested (non-plain) flows are not supported yet")
        pads = _shared_pads(z)
        # use-once (see above): the adjoint run below holds its own references
        shared.clear()
        K = compiled
        if not listed:
            c = identity_flow_scale(output)
            if c is not None:
                K = _compiled_for(c)
        pre = None
        if pads and _padding_domain(K) is fwd_domain:
            # forward values: field n_off + i of the adjoint layout
            read = K.read_fields
            pre = {n_off + i: P for i, P in pads.items() if (n_off + i) in read}
        if pre:
            K(*(slots + ns.psi() + full), padded=pre)
        else:
            K(*(slots + ns.psi() + full))
        return slots

    def _backward(z):
        if n_comp == 0:
            # no gradient-carrying input is referenced
            return
        if not ns.nested():
            # plain flows: the compiled adjoint kernel on plain slot lattices
            slots = run_adj_plain(z)
            for ci, r in slot_of.items():
                accum(children[ci - m], slots[r], 1)
            return
        # nested: the adjoint is a single-pass stencil, so the backward IS
        # that stencil acting on nodes again -- the recursion, with no
        # cshift.  A's fields are [slots] + [the m output flows] + [the n
        # forward values]; the flows become A's children, so differentiating
        # a slot runs the adjoint-of-adjoint stencil, and so on up the tower.
        A_output = g.ad.reverse.node(ns.lattices(n_comp))
        # z.value is a list of m forward outputs for a list-node target but
        # a single lattice/node for an m=1 target -- normalize to a list of
        # m entries (never `list(z.value)` on a bare field, which would
        # iterate its sites)
        if z.value is None:
            z_vals = [dummy] * m
        elif not listed:
            z_vals = [z.value]
        else:
            z_vals = list(z.value)
        # the adjoint graph lives ONE LEVEL DOWN: its operands are the
        # children's values (the inner nodes), exactly as node.__mul__
        # backpropagates with value_of(y).  The gradients still accumulate
        # into the children themselves (below).
        A_inputs = list(ns.psi()) + z_vals + [value_of(c) for c in children]
        A = matrix(compiled, A_output, *A_inputs)
        for ci, r in slot_of.items():
            accum(children[ci - m], A[r], 1)

    return ns.install(
        run_fwd,
        _backward,
        f"stencil({m} output, {len(inner.points)} points, {len(inner.code)} lines of code)",
    )
