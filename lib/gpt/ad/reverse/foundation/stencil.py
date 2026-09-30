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
# constant/temp).  The inputs occupy fields m..n-1.
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
# For a temp-free code the adjoint reads only the (never-written) forward
# values and output flows, never a flow slot, so all of its entries fuse to a
# single stage: the backward is ALSO a single kernel pass.  Codes with temps
# read temp-version slots at non-zero shift and split into stages (one kernel
# per stage, run in order on persistent slot lattices) because a compiled
# kernel snapshots non-zero-shift reads at call start.
#
# In a nested (multi-deep) pass the flows are lazy node graphs and cannot be
# fed to the kernels.  For a temp-free code the backward is the ADJOINT
# STENCIL acting on nodes again: it is built as a first-class stencil node
# whose forward runs the compiled adjoint kernel (on plain-resolved operands)
# and whose backward is the adjoint-of-adjoint stencil node.  So the recursion
# is a self-similar tower of stencils -- S, A = adjoint(S), A' = adjoint(A),
# ... -- with NO cshift: the shifts live in the compiled kernel's points, and
# the flow is a child of the adjoint node (its inputs are the output flows,
# the output values, and the inputs).  A code with temps reads temp-version
# slots, so its adjoint is not a valid forward stencil; the backward then
# falls back to interpreting the adjoint entries in the node domain (products
# of shifted (node cshift) and adjointed (node adj) fields) in code order.
#
# Because the m outputs are one list node (not m siblings), there is no
# fused multi-output bookkeeping: one node, one flow list, one adjoint run.
#
# Supported regime (the standard GPT stencil pattern: staple,
# parallel_transport_matrix, ...): factors only reference input fields (not
# outputs or temps); the first write of each output is fresh (accumulate=-1,
# or adds an already-valid input/temp), so an output's old value is not part
# of the computation; rewrites accumulate into the running value.
#
import weakref
import gpt as g
from gpt.ad.reverse.util import value_of, is_node, accum, identity_flow_scale


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


def adjoint_code(code, n_fields, outputs, ndim, temps=()):
    # The gradient of a stencil, in closed form, as another stencil.
    #
    #   code      : forward entries (target, accumulate, weight, factors),
    #               factors = [(field, point_tuple, adj_flag), ...]
    #   n_fields  : number of fields the forward stencil operates on
    #   outputs   : field indices that are stencil outputs (their flows are
    #               supplied by the caller)
    #   ndim      : point dimensionality
    #   temps     : field indices written internally (their flows are
    #               computed but not returned)
    #
    # Entry by entry, with phi the flow of the entry's target (the supplied
    # output flow for an output target, the version slot for a temp target)
    # and entry value z = weight * A_0 * ... * A_{k-1} (A_m = factor m,
    # possibly shifted and adjointed):
    #
    #   - flow of factor m  =  weight' * (the other A_l, in reverse order,
    #       each shifted by p_l - p_m relative to factor m) * (phi shifted
    #       by -p_m), with weight' = conj(weight) and the adjoint flag of
    #       every other factor flipped if A_m was not adjointed, and both
    #       unchanged if it was.  (The adjoint of a shifted/adjointed matrix
    #       is again a shifted/adjointed matrix -- the same rule as the
    #       gradient of g.cshift.)
    #   - flow of the accumulate read = phi, and for a temp target the
    #       previous version's flow receives the current version's flow (the
    #       running value is built on it).
    #
    # Field layout of the adjoint code:
    #
    #     [slot_0 ... slot_{n_comp-1}] + [psi_t for t in outputs]
    #     + [value_i for i in range(n_fields)]
    #
    #   slots  : one running flow slot per input field, then one per temp
    #            version (every temp writer creates a new version; readers
    #            -- acc reads and self-acc rewrites -- see the version of the
    #            last earlier writer).  All slots are targets of the adjoint
    #            code and zero-initialized by the caller.
    #   psi    : the supplied flow of each output (pure inputs)
    #   values : the forward values (pure inputs)
    #
    # Stages: a compiled kernel call snapshots non-zero-shift reads at the
    # start of the call (they see only earlier calls) and live-reads
    # zero-shift reads and accumulates in code order, so an entry may share a
    # call with the writers of the slots it live-reads (as long as they
    # precede it in code order) but must run in a strictly later call than the
    # writers of the slots it reads with a non-zero shift.  The stage of an
    # entry is the longest such dependency path (weight 1 per call barrier,
    # weight 0 per live, code-ordered dependency).  A temp-free code reads no
    # slots, so it fuses to a single stage (one kernel pass).  The stage split
    # is only needed for the compiled (plain) run; the node-domain run
    # interprets the entries in code order, which is live throughout.
    #
    # returns (entries, computed, outs) where
    #   entries  : [(stage, (target_slot, accumulate, weight, flist)), ...]
    #              in code order
    #   computed : slot order = [inputs..., "v:<temp>:<version>"...]
    #   outs     : the output field indices
    zero = (0,) * ndim
    outs = sorted(set(outputs))
    tmps = sorted(set(temps))
    inputs = [i for i in range(n_fields) if i not in outs and i not in tmps]

    # temp versions: every writer of a temp creates a new version of its
    # value; readers (acc reads, and self-acc rewrites) read the version of
    # the last writer before them.  Each version has its own flow slot.
    writers = {s: [i for i in range(len(code)) if code[i][0] == s] for s in tmps}
    for s in tmps:
        assert len(writers[s]) > 0, "temp %d is never written" % s

    # slot assignment: input flows first, then one slot per temp version
    computed = list(inputs)
    for s in tmps:
        computed.extend("v:%d:%d" % (s, k) for k in range(len(writers[s])))
    slot = {}
    for r, c in enumerate(computed):
        if isinstance(c, int):
            slot[("in", c)] = r
    for s in tmps:
        for k in range(len(writers[s])):
            slot[("v", s, k)] = computed.index("v:%d:%d" % (s, k))
    n_comp = len(computed)
    n_out = len(outs)
    PSI = lambda t: n_comp + outs.index(t)
    F = lambda i: n_comp + n_out + i

    entries = []
    slot_reads = []
    first_write = set()

    def emit(target, weight, flist):
        # the first write to a slot is fresh (the slot is zero-initialized by
        # the caller); subsequent writes accumulate.  This keeps the adjoint
        # code a valid forward stencil (the first write of an output does not
        # read its own old value)
        acc = -1 if target not in first_write else target
        first_write.add(target)
        entries.append((target, acc, weight, flist))
        # reads of slots (written by emitted entries): (slot, live), where
        # live means zero shift (in-place read, see the stage rule above);
        # psi and value fields are never written, no dependency
        slot_reads.append([(f, p == zero) for (f, p, a) in flist if f < n_comp])
        return len(entries) - 1

    for i in range(len(code) - 1, -1, -1):
        target, acc, weight, factors = code[i]
        k = len(factors)
        if target in tmps:
            kv = writers[target].index(i)
            phi = (slot[("v", target, kv)], 0, 0)
        else:
            phi = (PSI(target), 0, 0)
        for m in range(k):
            i_m, p_m, a_m = factors[m]

            def rel(l, p_m=p_m):
                return tuple(x - y for x, y in zip(factors[l][1], p_m))

            phi_ref = (phi[0], tuple(-x for x in p_m), a_m)
            if a_m == 0:
                flist = [(F(factors[l][0]), rel(l), 1 - factors[l][2]) for l in range(m - 1, -1, -1)]
                flist += [phi_ref]
                flist += [(F(factors[l][0]), rel(l), 1 - factors[l][2]) for l in range(k - 1, m, -1)]
                w = weight.conjugate()
            else:
                flist = [(F(factors[l][0]), rel(l), factors[l][2]) for l in range(m + 1, k)]
                flist += [phi_ref]
                flist += [(F(factors[l][0]), rel(l), factors[l][2]) for l in range(m)]
                w = weight
            emit(slot[("in", i_m)], w, flist)
        if acc != -1 and acc != target:
            if acc in inputs:
                emit(slot[("in", acc)], 1.0, [(phi[0], zero, 0)])
            elif acc in tmps:
                # read the version of the last writer before this entry
                ks = [kk for kk in range(len(writers[acc])) if writers[acc][kk] < i]
                assert ks, "acc reads temp %d before it is written" % acc
                emit(slot[("v", acc, ks[-1])], 1.0, [(phi[0], zero, 0)])
        if target in tmps and acc == target:
            kv = writers[target].index(i)
            if kv > 0:
                # self-acc handoff: flow of the previous version receives
                # the flow of this version
                emit(slot[("v", target, kv - 1)], 1.0, [(phi[0], zero, 0)])

    # stage assignment: longest dependency path over the emitted entries
    # (see the stage rule in the docstring)
    writers_of = {}
    for j, e in enumerate(entries):
        writers_of.setdefault(e[0], []).append(j)
    stage = [0] * len(entries)
    changed = True
    iters = 0
    while changed:
        changed = False
        iters += 1
        assert iters <= len(entries) + 1, "adjoint stage dependency cycle"
        for j in range(len(entries)):
            for (s, live) in slot_reads[j]:
                for i in writers_of.get(s, ()):
                    if i == j:
                        continue
                    need = stage[i] + (0 if (live and i < j) else 1)
                    if stage[j] < need:
                        stage[j] = need
                        changed = True

    entries = [[stage[j], e] for j, e in enumerate(entries)]
    return entries, computed, outs


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

    def flows(weight, fl, phi, fmap):
        # the product rule for one entry (see adjoint_code): per factor m, the
        # weight and factor list of its flow
        k = len(fl)
        res = []
        for mm in range(k):
            i_m, p_m, a_m = fl[mm]
            rel = lambda l, p_m=p_m: tuple(x - y for x, y in zip(fl[l][1], p_m))
            phi_ref = (phi, tuple(-x for x in p_m), a_m)
            if a_m == 0:
                f = [(fmap(fl[l][0]), rel(l), 1 - fl[l][2]) for l in range(mm - 1, -1, -1)]
                f += [phi_ref]
                f += [(fmap(fl[l][0]), rel(l), 1 - fl[l][2]) for l in range(k - 1, mm, -1)]
                w = complex(weight).conjugate()
            else:
                f = [(fmap(fl[l][0]), rel(l), fl[l][2]) for l in range(mm + 1, k)]
                f += [phi_ref]
                f += [(fmap(fl[l][0]), rel(l), fl[l][2]) for l in range(mm)]
                w = weight
            res.append((i_m, w, f))
        return res

    A, B_fresh, B_acc = [], [], []
    written_A, written_B = set(), set()
    # stage A: recompute the temporaries (kernel-local)
    for (t, acc, w, fl) in code:
        if t in tmps:
            A.append((LOC(t), -1 if acc == -1 else LOC(t), w, [(FA(f), p, a) for (f, p, a) in fl]))
    for (t, acc, w, fl) in reversed(code):
        if t in outs:
            for (i_m, wf, f) in flows(w, fl, PSI(t), FA):
                target = LAM(i_m) if i_m in tmps else SLOT(i_m)
                A.append((target, target if target in written_A else -1, wf, f))
                written_A.add(target)
        else:
            for (i_m, wf, f) in flows(w, fl, LAM(t), VAL_B):
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


def _matrix_local_temporaries(stencil, inner, fields):
    # node mode for a stencil with kernel-owned local temporaries (see
    # adjoint_code_local): the forward is the compiled kernel as always, the
    # backward is stage A (with local temporaries) followed by stage B
    points = inner.points
    temps = list(inner.temporaries)
    raw = [
        (e["target"], e["accumulate"], e["weight"],
         [(f, points[p], a) for (f, p, a) in e["factor"]])
        for e in inner.code
    ]
    output = fields[0]
    assert is_node(output), "stencil node mode: the output must be a node"
    m = len(output) if output._container.tag[0] is list else 1
    grid = output.grid
    otype_t = output.otype

    children = []
    for arg in fields[1:]:
        if is_node(arg) and arg._container.tag[0] is list:
            children.extend(arg[i] for i in range(len(arg)))
        elif is_node(arg):
            children.append(arg)
        else:
            children.append(g.ad.reverse.node_base(arg, with_gradient=False))

    # code index of each passed field (the temporaries are not passed)
    n_passed = m + len(children)
    full = [i for i in range(n_passed + len(temps)) if i not in temps]
    assert len(full) == n_passed
    outputs = full[:m]
    inputs = full[m:]
    for (tt, ac, w, fl) in raw:
        assert tt in outputs or tt in temps, "stencil node mode: inputs must not be written"
    referenced = {f for (tt, ac, w, fl) in raw for (f, p, a) in fl if f in inputs}

    cache = getattr(stencil, "_node_adj_local", None)
    if cache is None:
        cache = {}
        stencil._node_adj_local = cache
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

    def _nested():
        node_vals = [is_node(value_of(c)) for c in children if c.with_gradient]
        assert all(node_vals) or not any(node_vals), (
            "stencil node mode: gradient-carrying children must have uniform node depth")
        return any(node_vals)

    def _psi():
        output.materialize_gradient()
        return output.gradient if m > 1 else [output.gradient]

    def run_fwd():
        ops = []
        for c in children:
            v = value_of(c)
            if isinstance(v, g.expr):
                v = g(v)
            ops.append(v)
        if any(is_node(v) for v in ops):
            # nested: the value is this stencil one level down (see matrix)
            inner_out = [g.lattice(grid, otype_t) for _ in range(m)]
            inner_out = g.ad.reverse.node(inner_out if m > 1 else inner_out[0])
            return matrix(stencil, inner_out, *ops)
        outs = [g.lattice(grid, otype_t) for _ in range(m)]
        stencil(*(outs + ops))
        return outs[0] if m == 1 else outs

    def _backward(z):
        vals = [value_of(c) for c in children]
        psi = list(_psi())
        if not _nested():
            slots = [g.lattice(grid, otype_t) for _ in range(nI + nT)]
            for r in unwritten_A:
                slots[r][:] = 0
            K = KA
            if m == 1:
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
        A_out = g.ad.reverse.node([g.lattice(grid, otype_t) for _ in range(nI + nT)])
        Anode = matrix(KA, A_out, *psi, *vals)
        Bnode = None
        if KB_fresh is not None:
            B_out = g.ad.reverse.node([g.lattice(grid, otype_t) for _ in range(nI)])
            Bnode = matrix(KB_fresh, B_out, *[Anode[nI + t] for t in range(nT)], *vals)
        for k, c in enumerate(children):
            if not (c.with_gradient and inputs[k] in referenced):
                continue
            flow = Anode[k] if k in written_A else None
            if Bnode is not None and k in written_B:
                flow = Bnode[k] if flow is None else flow + Bnode[k]
            if flow is not None:
                accum(c, flow, 1)

    output._forward = run_fwd
    output._children = children
    output._backward = _backward
    output.value = None
    output.gradient = None
    output._tag = f"stencil({m} output, {len(temps)} local temporaries, {len(inner.code)} lines of code)"
    # the backward reads the inputs, never the output value
    output._reads_children = None
    output._reads_self = False
    return output


def matrix(stencil, *fields):
    inner = getattr(stencil, "local_stencil", stencil)
    if getattr(inner, "temporaries", ()):
        return _matrix_local_temporaries(stencil, inner, fields)
    points = inner.points
    raw = [
        (e["target"], e["accumulate"], e["weight"],
         [(f, points[p], a) for (f, p, a) in e["factor"]])
        for e in inner.code
    ]
    # highest field index the code touches
    fidx = set()
    for (tt, ac, w, fl) in raw:
        fidx.add(tt)
        if ac != -1:
            fidx.add(ac)
        fidx.update(f for (f, p, a) in fl)

    # the output is field(s) 0..m-1: a single node (m=1) or a list node (m)
    output = fields[0]
    assert is_node(output), "stencil node mode: the output must be a node"
    m = len(output) if output._container.tag[0] is list else 1
    grid = output.grid
    ndim = grid.nd
    otype_t = output.otype

    # the inputs occupy fields m..n_fields-1; expand each argument (a list
    # node to one child per element, a plain temp to a constant node)
    children = []
    for arg in fields[1:]:
        if is_node(arg) and arg._container.tag[0] is list:
            children.extend(arg[i] for i in range(len(arg)))
        elif is_node(arg):
            children.append(arg)
        else:
            children.append(g.ad.reverse.node_base(arg, with_gradient=False))

    # the field count comes from the ARGUMENTS, not from the highest index the
    # code happens to reference: a caller may legitimately pass fields the code
    # never reads (g.parallel_transport hands over all links, whatever
    # directions the paths use), exactly as the compiled kernel allows
    n_fields = m + len(children)
    assert max(fidx) < n_fields, (
        "stencil node mode: the code references field %d but only fields "
        "0..%d were passed" % (max(fidx), n_fields - 1))

    # field classification: outputs are 0..m-1, the other targets are temps,
    # the rest are inputs.  An output the code never writes is valid (it stays
    # at its zero-initialized value) -- this happens for derived adjoint
    # stencils, whose inputs include forward values the adjoint code never
    # references.
    targets = sorted({e[0] for e in raw})
    outputs = list(range(m))
    temps = [t for t in targets if t >= m]
    inputs = [i for i in range(n_fields) if i not in targets]
    for t in temps:
        assert not children[t - m].with_gradient, (
            "stencil node mode: temp field %d must be a constant" % t)

    # regime: factors reference only input fields; the first write of each
    # written output is fresh (does not read its own old value)
    first = {}
    for (tt, ac, w, fl) in raw:
        first.setdefault(tt, ac)
        for (f, p, a) in fl:
            assert f not in targets, (
                "stencil node mode: factors must reference input fields")
    for t in outputs:
        if t in targets:
            assert first[t] != t, (
                "stencil node mode: first write of output %d must not read its "
                "own old value (acc=-1, or acc=input/temp)" % t)

    referenced = set()
    for (tt, ac, w, fl) in raw:
        referenced.update(f for (f, p, a) in fl)
        if ac != -1 and ac != tt:
            referenced.add(ac)
    referenced = {i for i in referenced if i in inputs}

    # node depth: gradient-carrying children must be uniform (constants such
    # as temps are plain at any depth and carry no inner dependency).  This is
    # decided in the backward pass, not here: `value_of` evaluates a computed
    # child, and a value cached before the graph is first run would then be
    # reused by node.forward (which only recomputes values that are None)
    # instead of being rebuilt from the updated leaves.
    def _nested():
        node_vals = [is_node(value_of(c)) for c in children if c.with_gradient]
        assert all(node_vals) or not any(node_vals), (
            "stencil node mode: gradient-carrying children must have uniform "
            "node depth"
        )
        return any(node_vals)

    # the adjoint code, in closed form (another stencil): entries in code
    # order, its field layout, and the per-stage split for the compiled
    # (plain) run.  Cached on the stencil object, keyed by the output count
    # and temps (which the caller's arguments determine).
    cache = getattr(stencil, "_node_adj", None)
    if cache is None:
        cache = {}
        stencil._node_adj = cache
    key = (m, tuple(temps))
    if key not in cache:
        entries, computed, _outs = adjoint_code(
            raw, n_fields, outputs=tuple(outputs), ndim=ndim, temps=tuple(temps))
        levels = {}
        for lv, e in entries:
            levels.setdefault(lv, []).append(e)
        compiled = []
        n_comp = len(computed)
        level_codes = [levels[lv] for lv in sorted(levels)]
        for lvl in level_codes:
            pts = sorted({(0,) * ndim} | {p for (tt, ac, w, fl) in lvl for (f, p, a) in fl})
            pm = {p: i for i, p in enumerate(pts)}
            ccode = [(tt, ac, w, [(f, pm[p], a) for (f, p, a) in fl]) for (tt, ac, w, fl) in lvl]
            written = sorted({tt for (tt, ac, w, fl) in lvl})
            K = g.stencil.matrix(g.lattice(grid, otype_t), pts, ccode, cse=cse)
            n_adj_fields = n_comp + len(_outs) + n_fields
            # the fields read as factors (the dummy entries of the output
            # layout and unreferenced fields are not read, so the padded
            # stencil need not copy them)
            read = sorted({f for (tt, ac, w, fl) in lvl for (f, p, a) in fl})
            K.data_access_hints(written, read, [])
            compiled.append(K)
        # code-ordered entries with point tuples for the node-domain path
        adj_entries = [e for (_lv, e) in entries]
        # slots no stage writes stay at zero (every written slot's first
        # write is fresh, see emit, so only these need a zero fill)
        written_any = {e[0] for (_lv, e) in entries}
        unwritten = [r for r in range(n_comp) if r not in written_any]
        cache[key] = (computed, compiled, adj_entries, _outs, unwritten, level_codes)
    computed, compiled, adj_entries, _outs, unwritten, level_codes = cache[key]

    def _compiled_for(c):
        # the adjoint kernels for a flow c * identity (m = 1, see seedless_code)
        def build(c):
            ks = []
            for lvl in level_codes:
                code = seedless_code(lvl, len(computed), c)
                K = _compile(grid, otype_t, code)
                ks.append(K)
            return ks

        return _seedless_kernels(cache, key, c, build)
    n_comp = len(computed)
    n_off = n_comp + len(_outs)
    # slot position of each input field (computed = [inputs..., temp versions...])
    slot_of = {c: r for r, c in enumerate(computed) if isinstance(c, int)}
    zero = (0,) * ndim
    # the adjoint never reads an output's forward value (the regime asserts
    # factors reference inputs and first writes are fresh), but the kernel
    # padding plan needs a valid lattice at every read slot
    dummy = g.lattice(grid, otype_t)

    # Padded inputs shared between the forward run and the adjoint run.
    # A padded (multi-direction) stencil copies each input into a halo-padded
    # field; the adjoint reads the same forward values, so when both kernels
    # run on the same padding domain, the forward's padded inputs are handed
    # to the adjoint kernels instead of being copied again.  Invalidation:
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
        if m > 1:
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

    def _psi():
        # the m output flows; a single output node has one flow, a list node
        # a list of m (outputs that received no flow are zero)
        output.materialize_gradient()
        return output.gradient if m > 1 else [output.gradient]

    def run_fwd():
        # forward: one kernel pass computes all m outputs.  The operands are
        # resolved exactly ONE level down (value_of, not all the way to plain):
        # a lazy expr is materialized because the kernel needs lattices.
        ops = []
        for c in children:
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
            inner = [g.lattice(grid, otype_t) for _ in range(m)]
            inner = g.ad.reverse.node(inner if m > 1 else inner[0])
            return matrix(stencil, inner, *ops)

        outs = [g.lattice(grid, otype_t) for _ in range(m)]
        if (
            share_padded
            and fwd_domain is not None
            and any(_padding_domain(K) is fwd_domain for K in compiled)
        ):
            # keep the padded inputs for the adjoint run, see _keep_padded
            raw = [value_of(c) for c in children]
            _keep_padded(outs, raw, stencil(*(outs + ops)))
        else:
            stencil(*(outs + ops))
        return outs[0] if m == 1 else outs

    def run_adj_plain(z):
        # backward, plain flows: the compiled adjoint kernel(s), one per
        # stage (a single pass for a temp-free code), on persistent slot
        # lattices (the first write of every written slot is fresh, so only
        # the slots no stage writes are zero-filled)
        slots = [g.lattice(grid, otype_t) for _ in range(n_comp)]
        for r in unwritten:
            slots[r][:] = 0
        full = [dummy] * m  # output forward values are not read
        for c in children:
            v = value_of(c)
            if is_node(v):
                raise NotImplementedError(
                    "stencil node mode: nested (non-plain) values are not supported yet")
            full.append(v)
        for fl in _psi():
            if is_node(fl):
                raise NotImplementedError(
                    "stencil node mode: nested (non-plain) flows are not supported yet")
        pads = _shared_pads(z)
        # use-once (see above): the adjoint run below holds its own references
        shared.clear()
        kernels = compiled
        if m == 1:
            c = identity_flow_scale(output)
            if c is not None:
                kernels = _compiled_for(c)
        for K in kernels:
            pre = None
            if pads and _padding_domain(K) is fwd_domain:
                # forward values: field n_off + i of the adjoint layout
                read = K.read_fields
                pre = {n_off + i: P for i, P in pads.items() if (n_off + i) in read}
            if pre:
                K(*(slots + _psi() + full), padded=pre)
            else:
                K(*(slots + _psi() + full))
        return slots

    def _mul(a, b):
        # node-aware multiply, node-first (plain * node is not dispatchable)
        return a * b if is_node(a) else b * a if is_node(b) else a * b

    def _add(a, b):
        # node-aware add, node-first; plain + plain is a lazy expr, which
        # the plain-world slots must not become
        return a + b if is_node(a) else b + a if is_node(b) else g(a + b)

    def _plain(x):
        # keep plain operands materialized (g.adj of a plain is an expr)
        if is_node(x) or not isinstance(x, g.expr):
            return x
        return g(x)

    def _shift(x, p):
        # shift by a point; multi-direction points are composed
        for d in range(ndim):
            if p[d] != 0:
                x = _plain(g.cshift(x, d, p[d]))
        return x

    def run_adj_nodes():
        # backward, node flows: interpret the adjoint entries in code order,
        # one level down; each entry evaluates to a stencil-shaped node
        # expression (product of shifted/adjointed node fields) accumulated
        # into its slot.  This is the recursive step.
        vals = [None] * n_fields
        for k, c in enumerate(children):
            vals[m + k] = value_of(c)
        psi = _psi()
        slots = [None] * n_comp
        for (tt, ac, w, fl) in adj_entries:
            # the first write to a slot is fresh (ac=-1, the slot starts
            # zero/None so nothing is added back); later writes accumulate
            assert ac == tt or ac == -1, (
                "stencil node mode: adjoint entries must self-accumulate or be fresh")
            # adjoint layout: [slots] + [output flows] + [forward values]
            prod = None
            for (f, p, a) in fl:
                x = slots[f] if f < n_comp else (psi[f - n_comp] if f < n_off else vals[f - n_off])
                if x is None:
                    prod = None  # zero factor: the entry contributes nothing
                    break
                if p != zero:
                    x = _shift(x, p)
                if a:
                    x = _plain(g.adj(x))
                prod = x if prod is None else _mul(prod, x)
            if prod is None:
                continue
            tval = _mul(prod, w)
            if ac != -1 and slots[tt] is not None:
                tval = _add(tval, slots[tt])
            slots[tt] = tval
        return slots

    def _backward(z):
        nested = _nested()
        if not nested:
            # plain flows: the compiled adjoint kernel(s) on plain slot lattices
            slots = run_adj_plain(z)
            for k, c in enumerate(children):
                ci = m + k
                if c.with_gradient and ci in referenced:
                    s = slots[slot_of[ci]]
                    if s is not None:
                        accum(c, s, 1)
        elif not temps:
            # temp-free nested: the adjoint is a single-pass stencil, so the
            # backward IS that stencil acting on nodes again -- the recursion,
            # with no cshift.  A's fields are [slots] + [the m output flows]
            # + [the n forward values]; the flows become A's children, so
            # differentiating a slot runs the adjoint-of-adjoint stencil, and
            # so on up the tower.
            A_output = g.ad.reverse.node(
                [g.lattice(grid, otype_t) for _ in range(n_comp)])
            # z.value is a list of m forward outputs for a list-node target but
            # a single lattice/node for an m=1 target -- normalize to a list of
            # m entries (never `list(z.value)` on a bare field, which would
            # iterate its sites)
            if z.value is None:
                z_vals = [dummy] * m
            elif m == 1:
                z_vals = [z.value]
            else:
                z_vals = list(z.value)
            # the adjoint graph lives ONE LEVEL DOWN: its operands are the
            # children's values (the inner nodes), exactly as node.__mul__
            # backpropagates with value_of(y).  The gradients still
            # accumulate into the children themselves (below).
            A_inputs = list(_psi()) + z_vals + [value_of(c) for c in children]
            A = matrix(compiled[0], A_output, *A_inputs)
            for i, c in enumerate(children):
                ci = m + i
                if c.with_gradient and ci in referenced:
                    accum(c, A[slot_of[ci]], 1)
        else:
            # temps: the adjoint reads temp-version slots, so it is not a
            # valid forward stencil; interpret it in the node domain (cshift)
            slots = run_adj_nodes()
            for k, c in enumerate(children):
                ci = m + k
                if c.with_gradient and ci in referenced:
                    s = slots[slot_of[ci]]
                    if s is not None:
                        accum(c, s, 1)

    output._forward = run_fwd
    output._children = children
    output._backward = _backward
    output.value = None
    output.gradient = None
    output._tag = f"stencil({m} output, {len(points)} points, {len(inner.code)} lines of code)"
    # the backward reads the inputs, never the output value (the nested and
    # padding-sharing paths accept a missing value)
    output._reads_children = None
    output._reads_self = False
    return output
