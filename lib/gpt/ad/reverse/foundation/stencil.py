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
from gpt.ad.reverse.primitive import primitive, has_node
from gpt.ad.reverse.util import container, is_node, identity_flow_scale, constant


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


def _cache(obj, name):
    # a dict cached on obj (the adjoint codes, kernels and primitives of a
    # stencil)
    cache = getattr(obj, name, None)
    if cache is None:
        cache = {}
        setattr(obj, name, cache)
    return cache


def _lattice_list(grid, otype, n):
    return container(list, container(g.lattice, grid, otype), n)


def _op(stencil, out, n_inputs):
    # the primitive of a compiled stencil for the output container out (a
    # lattice, or a list of m lattices) and n_inputs input fields
    cache = _cache(stencil, "_node_ops")
    key = (str(out), n_inputs)
    if key not in cache:
        cache[key] = _stencil_op(stencil, out, n_inputs)
    return cache[key].op


class _stencil_op:
    # A compiled stencil as a primitive: inputs (the fields m..n-1) -> the
    # output field(s) 0..m-1 (a lattice, or a list of m lattices).  Its plain
    # implementation is the kernel; its vjp is the adjoint stencil (adjoint_code
    # or, with local temporaries, the two stages of adjoint_code_local),
    # applied as a primitive again: with plain flows it runs the compiled
    # adjoint kernel, in a nested pass it is a stencil node one level down
    # whose vjp is the adjoint of the adjoint, and so on (a self-similar tower
    # of stencils, with no cshift: the shifts live in the kernels' points)
    def __init__(self, stencil, out, n_inputs):
        inner = getattr(stencil, "local_stencil", stencil)
        points = inner.points
        self.stencil = stencil
        self.raw = [
            (e["target"], e["accumulate"], e["weight"],
             [(f, points[p], a) for (f, p, a) in e["factor"]])
            for e in inner.code
        ]
        # (a list output keeps its list semantics with a single element,
        # e.g. the adjoint of a stencil with one flowed input)
        self.listed = out.tag[0] is list
        self.m = out.tag[2] if self.listed else 1
        elem = out.tag[1] if self.listed else out
        self.grid, self.otype = elem.get_grid(), elem.get_otype()
        self.n_inputs = n_inputs
        self.temps = list(getattr(inner, "temporaries", ()))
        self.fwd_domain = _padding_domain(stencil)
        # whether the adjoint kernels run on the forward's padding domain
        # (None: not known yet)
        self.pad_match = None
        container = lambda *c, **static: out.copy()
        if self.temps:
            self._setup_local()
            self.op = primitive(
                f"stencil({self.m} output, {len(self.temps)} local temporaries, {len(inner.code)} lines of code)",
                self._plain,
                container,
                joint_vjp=self._vjp_local,
            )
        else:
            self._setup()
            self.op = primitive(
                f"stencil({self.m} output, {len(inner.points)} points, {len(inner.code)} lines of code)",
                self._plain,
                container,
                joint_vjp=self._vjp,
                fwd=self._fwd if share_padded and self.fwd_domain is not None else None,
            )

    def lattices(self, n):
        return [g.lattice(self.grid, self.otype) for _ in range(n)]

    def psi(self, z):
        # the m output flows; a single output node has one flow, a list node
        # a list of m (outputs that received no flow are zero)
        z.materialize_gradient()
        return list(z.gradient) if self.listed else [z.gradient]

    def _run(self, inputs, padded=None):
        # one kernel pass computes all m outputs (lazy expressions are
        # materialized: the kernel needs lattices); returns the outputs and
        # the padded fields the kernel ran on (padded stencils only)
        ops = [g(v) if isinstance(v, g.expr) else v for v in inputs]
        outs = self.lattices(self.m)
        if padded:
            pads = self.stencil(*(outs + ops), padded=padded)
        else:
            pads = self.stencil(*(outs + ops))
        return (outs if self.listed else outs[0]), outs, pads

    def _plain(self, *inputs, padded=None):
        return self._run(inputs, padded)[0]

    # temp-free stencils (adjoint_code)
    def _setup(self):
        # regime: only outputs are written, factors reference only inputs, the
        # first write of each output does not read its own old value.  An output
        # the code never writes is valid (it stays at its initial value) -- this
        # happens for derived adjoint stencils, whose inputs include forward
        # values the adjoint code never references.  The field count comes
        # from the arguments, not from the highest index the code references: a
        # caller may pass fields the code never reads (g.parallel_transport
        # hands over all links, whatever directions the paths use)
        m, n_fields = self.m, self.m + self.n_inputs
        first = {}
        referenced = set()
        for (tt, ac, w, fl) in self.raw:
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
        self.referenced = referenced
        # the adjoint never reads an output's forward value (factors reference
        # inputs, first writes are fresh), but the kernel padding plan needs a
        # valid lattice at every read slot
        self.dummy = g.lattice(self.grid, self.otype)

    def _adjoint(self, flowed):
        # the adjoint code of the flowed inputs, in closed form (another
        # stencil), compiled; cached on the stencil object, keyed by the output
        # count and the flowed inputs
        cache = _cache(self.stencil, "_node_adj")
        key = (self.m, flowed)
        if key not in cache:
            code = adjoint_code(self.raw, list(range(self.m)), flowed, self.grid.nd)
            cache[key] = (code, _compile(self.grid, self.otype, code))
        return cache, key

    def _fwd(self, *inputs):
        # the plain value of a node, keeping the halo-padded copies of its
        # read-only inputs for the adjoint run (same padding domain), so they
        # are not copied again.  The residual belongs to ONE forward value and
        # is used by at most one adjoint run (see primitive); it is emptied
        # when that value dies (e.g. a forward-only pass releases it), and is
        # only used while every shared input's value is the object that was
        # padded (so leaf values replaced between calls are never served
        # stale; as for retained forward values in general, fields must not be
        # modified in place while the values live)
        value, outs, padded = self._run(inputs)
        pads = {}
        if padded is not None and self.pad_match is not False:
            read_only = set(self.stencil.read_fields) - set(self.stencil.write_fields)
            for i in sorted(self.referenced & read_only):
                try:
                    pads[i] = (weakref.ref(inputs[i - self.m]), padded[i])
                except TypeError:
                    continue
        if not pads:
            return value, None
        residual = {"pads": pads}
        residual["out"] = weakref.ref(outs[0], lambda _r, d=residual: d.clear())
        return value, residual

    def _shared_pads(self, z, values, residual, K, n_off):
        # the forward's padded inputs that are still valid for z, in the
        # adjoint kernel's layout (forward value i is field n_off + i)
        self.pad_match = _padding_domain(K) is self.fwd_domain
        if not residual or not self.pad_match:
            return None
        out = residual["out"]()
        v = z.value
        if self.listed and isinstance(v, list) and len(v) == self.m:
            v = v[0]
        if out is None or v is not out:
            return None
        read = K.read_fields
        pre = {}
        for i, (ref, P) in residual["pads"].items():
            if ref() is values[i - self.m] and (n_off + i) in read:
                pre[n_off + i] = P
        return pre or None

    def _vjp(self, z, needed, *values, residual=None):
        # the inputs whose flow is needed (referenced, gradient-carrying): the
        # adjoint computes only their slots, so constant inputs cost nothing
        m = self.m
        flowed = tuple(m + k for k in needed if m + k in self.referenced)
        if not flowed:
            return {}
        cache, key = self._adjoint(flowed)
        code, K = cache[key]
        n_comp = len(flowed)
        if not self.listed:
            c = identity_flow_scale(z)
            if c is not None:
                # a flow c * identity (see seedless_code)
                K = _seedless_kernels(
                    cache, key, c,
                    lambda c: _compile(self.grid, self.otype, seedless_code(code, n_comp, c)))
        # the adjoint's fields: [slots] + [the m output flows] + [the m forward
        # values, never read] + [the input values]
        args = self.psi(z) + [self.dummy] * m + list(values)
        static = {}
        if self.fwd_domain is not None and not has_node(args):
            pre = self._shared_pads(z, values, residual, K, n_comp + m)
            if pre:
                static["padded"] = pre
        A = _op(K, _lattice_list(self.grid, self.otype, n_comp), len(args))(*args, **static)
        return {i - m: A[r] for r, i in enumerate(flowed)}

    # stencils with local temporaries (adjoint_code_local)
    def _setup_local(self):
        m, temps = self.m, self.temps
        # code index of each passed field (the temporaries are not passed)
        n_passed = m + self.n_inputs
        full = [i for i in range(n_passed + len(temps)) if i not in temps]
        assert len(full) == n_passed
        self.outputs, self.inputs = full[:m], full[m:]
        for (tt, ac, w, fl) in self.raw:
            assert tt in self.outputs or tt in temps, "stencil node mode: inputs must not be written"
        self.referenced = {f for (tt, ac, w, fl) in self.raw for (f, p, a) in fl if f in self.inputs}
        cache = _cache(self.stencil, "_node_adj_local")
        key = (m, n_passed)
        if key not in cache:
            A, locA, B_fresh, B_acc, written_A, written_B = adjoint_code_local(
                self.raw, self.outputs, temps, self.inputs, self.grid.nd)
            KA = _compile(self.grid, self.otype, A, locA)
            KB_fresh = _compile(self.grid, self.otype, B_fresh) if B_fresh else None
            KB_acc = _compile(self.grid, self.otype, B_acc) if B_acc else None
            nI, nT = len(self.inputs), len(temps)
            # A's outputs that it never writes but stage B_acc or the caller reads
            unwritten_A = [r for r in range(nI + nT) if r not in written_A]
            cache[key] = (KA, KB_fresh, KB_acc, written_A, written_B, unwritten_A, A, locA)
        self.local_cache, self.local_key = cache, key

    def _vjp_local(self, z, needed, *values):
        # stage A (with local temporaries) followed by stage B (temp-free)
        KA, KB_fresh, KB_acc, written_A, written_B, unwritten_A, A_code, locA = self.local_cache[
            self.local_key]
        nI, nT = len(self.inputs), len(self.temps)
        needed = [k for k in needed if self.inputs[k] in self.referenced]
        psi = self.psi(z)
        if not has_node(psi) and not has_node(values):
            # plain: one fused run, stage B accumulating into the slots of A
            slots = self.lattices(nI + nT)
            for r in unwritten_A:
                slots[r][:] = 0
            K = KA
            if not self.listed:
                c = identity_flow_scale(z)
                if c is not None:
                    # a flow c * identity (m = 1, see seedless_code); psi is
                    # the field after the slots and the temporary flows
                    K = _seedless_kernels(
                        self.local_cache, self.local_key, c,
                        lambda c: _compile(
                            self.grid, self.otype, seedless_code(A_code, nI + nT, c), locA))
            K(*(slots + psi + list(values)))
            if KB_acc is not None:
                KB_acc(*(slots + list(values)))
            return {k: slots[k] for k in needed}
        # nested: both stages as stencil nodes one level down; stage A has
        # local temporaries, stage B is temp-free
        A = _op(KA, _lattice_list(self.grid, self.otype, nI + nT), len(psi) + len(values))(
            *psi, *values)
        B = None
        if KB_fresh is not None:
            B = _op(KB_fresh, _lattice_list(self.grid, self.otype, nI), nT + len(values))(
                *[A[nI + t] for t in range(nT)], *values)
        flows = {}
        for k in needed:
            flow = A[k] if k in written_A else None
            if B is not None and k in written_B:
                flow = B[k] if flow is None else flow + B[k]
            if flow is not None:
                flows[k] = flow
        return flows


def matrix(stencil, *fields):
    # stencil(output, *inputs) with an output node: the output node becomes
    # the computed node of the stencil's primitive.  Each input is a node, a
    # LIST node (one input field per element) or a plain value (a constant)
    output = fields[0]
    assert is_node(output), "stencil node mode: the output must be a node"
    children = []
    for arg in fields[1:]:
        if is_node(arg) and arg._container.tag[0] is list:
            children.extend(arg[i] for i in range(len(arg)))
        else:
            children.append(constant(arg))
    z = _op(stencil, output._container, len(children)).node(*children)
    for name in [
        "_forward",
        "_backward",
        "_children",
        "_tag",
        "_reads_children",
        "_reads_self",
        "with_gradient",
    ]:
        setattr(output, name, getattr(z, name))
    output.value = None
    output.gradient = None
    return output
