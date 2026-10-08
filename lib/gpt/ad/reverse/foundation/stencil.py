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
# constant).  The inputs occupy fields m..n-1; kernel-owned local temporaries
# (g.stencil.matrix(..., temporaries=[...])) are not passed.
#
# The output node is converted in place into a computed node whose
#   - forward is a SINGLE kernel pass computing all m outputs, and
#   - backward runs the ADJOINT of the stencil code, which is a stencil in
#     closed form (adjoint_code): the same entries with the product rule
#     applied to each factor (shifts negated/relativized, adjoint flags
#     adjusted, the m output flows as extra inputs), written into flow slots.
#     So, as for g.cshift -- whose gradient is again a cshift with the
#     displacement negated -- the gradient of a stencil is a stencil.  With
#     local temporaries the adjoint is two stencils: stage A (again with
#     local temporaries) and stage B (temp-free); without, stage A alone.
#
# The adjoint reads only the (never-written) forward values and output flows,
# never a flow slot, so each stage is a single kernel pass.
#
# In a recorded pass (create_graph) the flows are lazy node graphs and
# cannot be fed to the kernels.  The backward is then the ADJOINT STENCIL
# acting on nodes again: each stage is a stencil node (the primitive of its compiled
# kernel) whose backward is again derived by adjoint_code.  So the recursion
# is a self-similar tower of stencils -- S, A = adjoint(S), A' = adjoint(A),
# ... -- with NO cshift: the shifts live in the compiled kernels' points.
#
# Because the m outputs are one list node (not m siblings), there is no
# fused multi-output bookkeeping: one node, one flow list, one adjoint run.
#
# Supported regime (the standard GPT stencil pattern: staple,
# parallel_transport_matrix, ...): only outputs and local temporaries are
# written; outputs are never read; the first write of each output does not
# read its old value (accumulate=-1, or adds an input); rewrites accumulate
# into the running value.  Local temporaries (R1) are written and read at
# the zero point only, (R2) an entry that reads one reads all its factors at
# the zero point, are built from inputs only (no chains), are written before
# they are read, and are written fresh or accumulating into themselves.
#
import gpt as g
from gpt.ad.reverse.primitive import primitive, has_node
from gpt.ad.reverse.util import container, is_node, constant
from gpt.ad.reverse import flow as flows


# combine common subexpressions in the executed adjoint kernels (the minimal
# number of uses of a per-site temporary, see core/local_stencil/cse.py;
# False: off).  Only the execution plan changes: the adjoint codes the tower
# derives from are the uncombined ones.
cse = 2


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


def adjoint_code(code, outputs, inputs, temps, flowed, ndim):
    # The gradient of a stencil, in closed form, as stencils.
    #
    #   code    : forward entries (target, accumulate, weight, factors),
    #             factors = [(field, point_tuple, adj_flag), ...], in the
    #             supported regime (see the top of this file)
    #   outputs, inputs, temps : the code's field indices of each kind; the
    #             passed fields are the outputs and inputs (in index order)
    #   flowed  : the inputs whose flows are computed (the entries of the
    #             others are dropped)
    #   ndim    : point dimensionality
    #
    # Entry by entry, with psi the supplied flow of an output and lam_T the
    # flow of a temporary T: the flows of the factors by the product rule
    # (_product_rule), and the flow of an accumulate read of an input is psi.
    #   stage A (with local temporaries, if any): recomputes the temporaries
    #     and runs the reverse sweep over the entries writing outputs; by R2
    #     the flows of entries reading temporaries are local, so they go into
    #     the input slots and into lam_T (written to memory)
    #   stage B (temp-free; only with temporaries): the flows of the
    #     temporary-defining entries, which read lam_T at shifted points (a
    #     barrier after stage A)
    # Only the temporaries defined from a flowed input get a flow.  Both
    # stages are stencils of the supported kinds again, so the adjoint of the
    # adjoint is derived the same way (a self-similar tower).
    #
    # Field layouts (nS flowed inputs, nL temporaries with a flow, m outputs,
    # the passed fields in index order, nT temporaries):
    #   stage A: [slot]*nS [lam_T]*nL [psi]*m [value of each passed field]
    #            [local T]*nT  (the local T are kernel-owned, not passed)
    #   stage B: [slot]*nS [lam_T]*nL [value of each passed field]
    # Without temporaries stage A is [slot]*nS [psi]*m [values].  The values
    # of the outputs are never read (any field of the type takes their
    # place).  Returns a dict: the codes A (with its local temporaries locA),
    # B_fresh (slots start fresh: a separate contribution) and B_acc
    # (accumulates into the slots of A: one fused plain run), the slots each
    # stage writes, and the layout sizes.
    zero = (0,) * ndim
    outputs, temps = list(outputs), list(temps)
    passed = sorted(list(outputs) + list(inputs))
    flowed = sorted(flowed)
    lam_temps = [
        T
        for T in temps
        if any(t == T and any(f in flowed for (f, p, a) in fl) for (t, ac, w, fl) in code)
    ]
    nS, nL, m = len(flowed), len(lam_temps), len(outputs)
    SLOT = lambda i: flowed.index(i)
    LAM = lambda T: nS + lam_temps.index(T)
    PSI = lambda o: nS + nL + outputs.index(o)
    VAL_A = lambda f: nS + nL + m + passed.index(f)
    LOC = lambda T: nS + nL + m + len(passed) + temps.index(T)
    FA = lambda f: LOC(f) if f in temps else VAL_A(f)
    VAL_B = lambda f: nS + nL + passed.index(f)

    A, B_fresh, B_acc = [], [], []
    written_A, written_B = set(), set()

    def emit(entries, written, target, weight, flist):
        entries.append((target, target if target in written else -1, weight, flist))
        written.add(target)

    # stage A: recompute the temporaries (kernel-local)
    for (t, acc, w, fl) in code:
        if t in temps:
            A.append((LOC(t), -1 if acc == -1 else LOC(t), w, [(FA(f), p, a) for (f, p, a) in fl]))
    for (t, acc, w, fl) in reversed(code):
        if t in outputs:
            for (i_m, wf, f) in _product_rule(w, fl, PSI(t), FA):
                if i_m in lam_temps:
                    emit(A, written_A, LAM(i_m), wf, f)
                elif i_m in flowed:
                    emit(A, written_A, SLOT(i_m), wf, f)
            if acc not in (-1, t) and acc in flowed:
                emit(A, written_A, SLOT(acc), 1.0, [(PSI(t), zero, 0)])
        elif t in lam_temps:
            for (i_m, wf, f) in _product_rule(w, fl, LAM(t), VAL_B):
                if i_m in flowed:
                    B_acc.append((SLOT(i_m), SLOT(i_m), wf, f))
                    emit(B_fresh, written_B, SLOT(i_m), wf, f)
    return dict(
        A=A,
        locA=[LOC(T) for T in temps],
        B_fresh=B_fresh,
        B_acc=B_acc,
        written_A=written_A,
        written_B=written_B,
        nS=nS,
        nL=nL,
    )


def seedless_code(code, psi_index, c):
    # the code for a flow psi = c * identity (a scaled_identity flow, see flow.py):
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
    # (kernel, whether it still reads psi): the seedless code keeps psi only
    # in entries where it is the only factor; otherwise the flow field is
    # never built (any field of the type can take its place)
    variants = cache.setdefault(("seedless",) + key, {})
    if c not in variants:
        if len(variants) >= seedless_cache_size:
            variants.pop(next(iter(variants)))
        variants[c] = build(c)
    return variants[c]


def _seedless_compiled(grid, otype, code, psi_index, c, temporaries=()):
    code = seedless_code(code, psi_index, c)
    reads_psi = any(f == psi_index for (tt, ac, w, fl) in code for (f, p, a) in fl)
    return _compile(grid, otype, code, temporaries), reads_psi


def _compile(grid, otype, code, temporaries=()):
    # a compiled matrix stencil for code with point tuples
    ndim = grid.nd
    temps = sorted(temporaries)
    pts = sorted({(0,) * ndim} | {p for (tt, ac, w, fl) in code for (f, p, a) in fl})
    pm = {p: i for i, p in enumerate(pts)}
    ccode = [(tt, ac, w, [(f, pm[p], a) for (f, p, a) in fl]) for (tt, ac, w, fl) in code]
    return g.stencil.matrix(g.lattice(grid, otype), pts, ccode, temporaries=temps, cse=cse)


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
    # implementation is the kernel; its vjp is the adjoint (adjoint_code),
    # applied as primitives again: with plain flows one fused run of the
    # compiled adjoint kernels, in a recorded pass stencil nodes whose vjps
    # are the adjoints of the adjoint, and so on (a self-similar
    # tower of stencils, with no cshift: the shifts live in the kernels'
    # points)
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
        self.temps = sorted(getattr(inner, "temporaries", ()))
        self._setup(n_inputs)
        # the values of fields a kernel does not read (output values, a flow
        # the seedless kernels drop): any field of the type
        self.dummy = g.lattice(self.grid, self.otype)
        self.op = primitive(
            f"stencil({self.m} output, {len(points)} points, {len(self.temps)} local "
            f"temporaries, {len(inner.code)} lines of code)",
            self._plain,
            lambda *c, **static: out.copy(),
            joint_vjp=self._vjp,
        )

    def _setup(self, n_inputs):
        # the code's field indices of the passed fields (the temporaries are
        # not passed; the field count comes from the arguments, not from the
        # highest index the code references: a caller may pass fields the
        # code never reads, e.g. g.parallel_transport hands over all links)
        # and the regime checks (see the top of this file)
        m, temps = self.m, self.temps
        zero = (0,) * self.grid.nd
        full = [i for i in range(m + n_inputs + len(temps)) if i not in temps]
        self.outputs, self.inputs = full[:m], full[m:]
        outputs, inputs = set(self.outputs), set(self.inputs)
        first, first_read, referenced = {}, {}, set()
        for j, (t, acc, w, fl) in enumerate(self.raw):
            assert t in outputs or t in temps, (
                "stencil node mode: only outputs and local temporaries are written "
                "(field %d)" % t)
            reads_tmp = any(f in temps for (f, p, a) in fl)
            for (f, p, a) in fl:
                assert f in inputs or f in temps, (
                    "stencil node mode: factors reference inputs or local temporaries "
                    "(field %d)" % f)
                if f in temps:
                    assert p == zero, "local temporaries are read at the zero point (R1)"
                    first_read.setdefault(f, j)
                else:
                    referenced.add(f)
                if reads_tmp:
                    assert p == zero, (
                        "an entry reading a local temporary reads all its factors at "
                        "the zero point (R2)")
            if t in temps:
                assert not reads_tmp, "local temporaries built from temporaries are not supported"
                assert acc in (-1, t), (
                    "local temporaries are written fresh or accumulate into themselves")
            else:
                first.setdefault(t, acc)
                if acc not in (-1, t):
                    assert acc in inputs, (
                        "stencil node mode: accumulate reads an input field (field %d)" % acc)
                    referenced.add(acc)
        for j, (t, acc, w, fl) in enumerate(self.raw):
            if t in first_read:
                assert j < first_read[t], "all writes of a local temporary precede its reads"
        for t, acc in first.items():
            assert acc != t, (
                "stencil node mode: first write of output %d must not read its "
                "own old value (acc=-1, or acc=input)" % t)
        self.referenced = referenced
        # outputs the code never writes are zero (e.g. the flow of a local
        # temporary that no output entry reads, in a derived adjoint)
        written = {t for (t, acc, w, fl) in self.raw}
        self.unwritten = [k for k, o in enumerate(self.outputs) if o not in written]

    def lattices(self, n):
        return [g.lattice(self.grid, self.otype) for _ in range(n)]

    def psi(self, z):
        # the m output flows; a single output node has one flow, a list node
        # a list of m (outputs that received no flow are zero)
        z.materialize_gradient()
        return list(z.gradient) if self.listed else [z.gradient]

    def _plain(self, *inputs):
        # one kernel pass computes all m outputs (lazy expressions are
        # materialized: the kernel needs lattices)
        ops = [g(v) if isinstance(v, g.expr) else v for v in inputs]
        outs = self.lattices(self.m)
        for k in self.unwritten:
            outs[k][:] = 0
        self.stencil(*(outs + ops))
        return outs if self.listed else outs[0]

    def _adjoint(self, flowed):
        # the adjoint codes of the flowed inputs, compiled; cached on the
        # stencil object, keyed by the output count and the flowed inputs
        cache = _cache(self.stencil, "_node_adj")
        key = (self.m, flowed)
        if key not in cache:
            adj = adjoint_code(self.raw, self.outputs, self.inputs, self.temps, flowed, self.grid.nd)
            KA = _compile(self.grid, self.otype, adj["A"], adj["locA"])
            KB_fresh = _compile(self.grid, self.otype, adj["B_fresh"]) if adj["B_fresh"] else None
            KB_acc = _compile(self.grid, self.otype, adj["B_acc"]) if adj["B_acc"] else None
            cache[key] = (adj, KA, KB_fresh, KB_acc)
        return cache, key

    def _vjp(self, z, needed, *values):
        # the inputs whose flow is needed (referenced, gradient-carrying): the
        # adjoint computes only their slots, so constant inputs cost nothing
        flowed = tuple(self.inputs[k] for k in needed if self.inputs[k] in self.referenced)
        if not flowed:
            return {}
        cache, key = self._adjoint(flowed)
        adj, KA, KB_fresh, KB_acc = cache[key]
        nS, nL, m = adj["nS"], adj["nL"], self.m
        # (a scaled identity is a plain flow, also in a recorded pass, where
        # it is a constant; the seedless kernels are for plain values)
        c = None if self.listed or has_node(values) else flows.scale(z.flow)
        if c is None:
            psi = self.psi(z)
        else:
            # a flow c * identity (see seedless_code); psi is the field after
            # the slots and the flows of the temporaries
            KA, reads_psi = _seedless_kernels(
                cache, key, c,
                lambda c: _seedless_compiled(self.grid, self.otype, adj["A"], nS + nL, c, adj["locA"]))
            psi = [z.gradient if reads_psi else self.dummy]
        # the fields after the slots and temporary flows: [the m output flows]
        # + [the m forward values, never read] + [the input values]
        values = [self.dummy] * m + list(values)
        if not has_node(psi) and not has_node(values):
            # plain: one fused run, stage B accumulating into the slots of A
            slots = self.lattices(nS + nL)
            for r in range(nS + nL):
                if r not in adj["written_A"]:
                    slots[r][:] = 0
            KA(*(slots + psi + values))
            if KB_acc is not None:
                KB_acc(*(slots + values))
            return {k: slots[r] for r, k in enumerate(self._children_of(flowed))}
        # recorded: both stages as stencil nodes
        A = _op(KA, _lattice_list(self.grid, self.otype, nS + nL), len(psi) + len(values))(
            *psi, *values)
        B = None
        if KB_fresh is not None:
            B = _op(KB_fresh, _lattice_list(self.grid, self.otype, nS), nL + len(values))(
                *[A[nS + l] for l in range(nL)], *values)
        result = {}
        for r, k in enumerate(self._children_of(flowed)):
            flow = A[r] if r in adj["written_A"] else None
            if B is not None and r in adj["written_B"]:
                flow = B[r] if flow is None else flow + B[r]
            if flow is not None:
                result[k] = flow
        return result

    def _children_of(self, flowed):
        # the child (input position) of each flowed field
        return [self.inputs.index(i) for i in flowed]


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
    # (the output node is constructed anew: its old value is discarded)
    z = _op(stencil, output._container, len(children)).node(*children)
    output.value = None
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
    output.gradient = None
    return output
