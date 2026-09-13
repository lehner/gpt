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
import gpt as g
from gpt.ad.reverse.util import value_of, is_node, accum


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


def matrix(stencil, *fields):
    inner = getattr(stencil, "local_stencil", stencil)
    points = inner.points
    raw = [
        (e["target"], e["accumulate"], e["weight"],
         [(f, points[p], a) for (f, p, a) in e["factor"]])
        for e in inner.code
    ]
    # number of fields the code operates on
    fidx = set()
    for (tt, ac, w, fl) in raw:
        fidx.add(tt)
        if ac != -1:
            fidx.add(ac)
        fidx.update(f for (f, p, a) in fl)
    n_fields = max(fidx) + 1

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
    assert len(children) == n_fields - m, (
        "stencil node mode: the input arguments must expand to fields %d..%d "
        "(got %d)" % (m, n_fields - 1, len(children)))

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
    # as temps are plain at any depth and carry no inner dependency)
    node_vals = [is_node(value_of(c)) for c in children if c.with_gradient]
    nested = any(node_vals)
    assert all(node_vals) or not any(node_vals), (
        "stencil node mode: gradient-carrying children must have uniform "
        "node depth"
    )

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
        for lv in sorted(levels):
            lvl = levels[lv]
            pts = sorted({(0,) * ndim} | {p for (tt, ac, w, fl) in lvl for (f, p, a) in fl})
            pm = {p: i for i, p in enumerate(pts)}
            ccode = [(tt, ac, w, [(f, pm[p], a) for (f, p, a) in fl]) for (tt, ac, w, fl) in lvl]
            written = sorted({tt for (tt, ac, w, fl) in lvl})
            K = g.stencil.matrix(g.lattice(grid, otype_t), pts, ccode)
            n_adj_fields = n_comp + len(_outs) + n_fields
            K.data_access_hints(written, [i for i in range(n_adj_fields) if i not in written], [])
            compiled.append(K)
        # code-ordered entries with point tuples for the node-domain path
        adj_entries = [e for (_lv, e) in entries]
        cache[key] = (computed, compiled, adj_entries, _outs)
    computed, compiled, adj_entries, _outs = cache[key]
    n_comp = len(computed)
    n_off = n_comp + len(_outs)
    # slot position of each input field (computed = [inputs..., temp versions...])
    slot_of = {c: r for r, c in enumerate(computed) if isinstance(c, int)}
    zero = (0,) * ndim
    # the adjoint never reads an output's forward value (the regime asserts
    # factors reference inputs and first writes are fresh), but the kernel
    # padding plan needs a valid lattice at every read slot
    dummy = g.lattice(grid, otype_t)

    def _psi():
        # the m output flows; a single output node has one flow, a list node
        # a list of m
        return output.gradient if m > 1 else [output.gradient]

    def run_fwd():
        # forward: one kernel pass computes all m outputs
        outs = [g.lattice(grid, otype_t) for _ in range(m)]
        full = list(outs)
        for c in children:
            v = value_of(c)
            while is_node(v):
                v = value_of(v)
            full.append(v)
        stencil(*full)
        return outs[0] if m == 1 else outs

    def run_adj_plain():
        # backward, plain flows: the compiled adjoint kernel(s), one per
        # stage (a single pass for a temp-free code), on persistent
        # (zero-initialized) slot lattices
        slots = [g.lattice(grid, otype_t) for _ in range(n_comp)]
        for s in slots:
            s[:] = 0
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
        for K in compiled:
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
        if not nested:
            # plain flows: the compiled adjoint kernel(s) on plain slot lattices
            slots = run_adj_plain()
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
            A_inputs = list(_psi()) + z_vals + list(children)
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
    return output
