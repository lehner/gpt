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
# Node foundation for compiled matrix stencils.
#
# Calling a stencil (g.stencil.matrix / the matrix_padded wrapper) with node
# fields dispatches here: the target node(s) are converted in place into
# computed nodes whose
#   - forward runs the compiled kernel on the plain values, and
#   - backward runs the compiled ADJOINT stencil derived from the same code
#     (gpt.core.local_stencil.adjoint), one fused kernel per stage (see the
#     stage-assignment rules in adjoint_code).
#
# Current limitations (plain level only):
#   - a single output target field, which must be a node; other (temp) target
#     fields must be plain and are allocated by the caller
#   - the first write of the target must not read the target's own old value
#     (accumulate = -1, or acc = input/temp), so the scratch's old value is
#     not part of the computation
#   - the target's old value must be plain (no nested nodes)
#   - node values/flows must be plain when the kernels run (a 1-deep graph
#     reversed with plain flow); nested (multi-deep) use raises
#
import gpt as g
from gpt.ad.reverse.util import value_of, is_node, accum


def adjoint_code(code, n_fields, outputs, ndim, temps=()):
    # derive the code of the adjoint stencil(s) of `code` (see module
    # docstring for the per-entry formulas).
    #
    #   code      : forward entries (target, accumulate, weight, factors)
    #               with factors = [(field, point_tuple, adj_flag), ...]
    #   n_fields  : number of fields the forward stencil operates on
    #   outputs   : field indices that are stencil outputs (fresh values;
    #               their flows are supplied by the caller)
    #   ndim      : point dimensionality
    #   temps     : field indices that are written internally (their flows
    #               are computed but not returned)
    #
    # Supported regime (the standard GPT stencil pattern: staple,
    # parallel_transport_matrix, ...): factors only reference input fields;
    # the first write of a target uses accumulate=-1 (fresh write) or adds
    # an already-valid input/temp field (acc=input/temp); rewrites of a
    # target must accumulate into the running value (acc=target).  (See the
    # regime validation below; violations would create dead code.)
    #
    # Flow slots: every input and temp field has a *running* flow slot.
    # Output fields have no slot: their flow is supplied and read directly
    # (nothing writes an output's running value under the supported regime,
    # except the output entries themselves, whose backprops read the
    # supplied flow).
    #
    # Stages: entries are emitted in reverse forward order and grouped into
    # stages; the caller compiles one stencil per stage and runs them in
    # stage order with persistent slot lattices.  Within a stage, entries
    # execute in code order, per site.
    #
    # Slot reads come in two flavors (see the compiled kernel):
    #   - zero-shift factor reads and the in-place accumulate read are LIVE
    #     (they see earlier stages plus same-stage, earlier-in-code writes),
    #   - non-zero-shift factor reads are SNAPSHOTs staged at the start of
    #     the call (they see only earlier stages).
    # So an entry may share a stage with the writers of the slots it
    # live-reads (as long as they precede it in code order), but must run in
    # a strictly later stage than the writers of the slots it reads with a
    # non-zero shift.  The stage of an entry is the longest such dependency
    # path (weight 1 per stage barrier, weight 0 per live, code-ordered
    # dependency).  (Codes without temps, or where temp flows are only
    # consumed at zero shift, fuse to a single stage.)
    #
    # returns (entries, computed, outs) where
    #   entries  : list of (stage, (target_slot, accumulate, weight, flist))
    #              in code order
    #   computed : slot order = [inputs..., "v:<temp>:<version>"...]; the
    #              adjoint field layout is [slot_i for i in computed]
    #              + [psi_t for t in outs] + [f_i for i in range(n_fields)]
    #   stages are per entry as documented above; the caller runs one
    #              compiled stencil per stage, in stage order, with
    #              persistent (zero-initialized) slot lattices
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

    def emit(target, weight, flist):
        # every entry accumulates into its slot; the caller
        # zero-initializes the slot lattices
        entries.append((target, target, weight, flist))
        # reads of slots (written by emitted entries): (slot, live), where
        # live means zero shift (in-place read, see stage rules above);
        # psi and forward-value fields are never written, no dependency
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
    # (see the stage rules in the docstring)
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
    grid = fields[0].grid
    ndim = grid.nd
    n = len(fields)
    targets = sorted({e[0] for e in raw})
    node_targets = [x for x in targets if is_node(fields[x])]
    assert len(node_targets) == 1, (
        "stencil node mode: exactly one target (the output) must be a node")
    t = node_targets[0]
    temps = [x for x in targets if x != t]
    for x in temps:
        assert not is_node(fields[x]), "stencil node mode: temps must be plain"
    first = {}
    for (tt, ac, w, fl) in raw:
        first.setdefault(tt, ac)
        for (f, p, a) in fl:
            assert f not in targets, (
                "stencil node mode: factors must reference input fields")
    assert first[t] != t, (
        "stencil node mode: first write of the target must not read its own "
        "old value (acc=-1, or acc=input/temp)")

    tv = value_of(fields[t])
    assert not is_node(tv), "stencil node mode: target's old value must be plain"
    otype_t = tv.otype

    inputs = [i for i in range(n) if i != t]
    # plain operands (e.g. temps) are promoted to constant nodes; note that
    # nodify on a single plain argument passes through unwrapped
    children = [
        fields[i] if is_node(fields[i]) else g.ad.reverse.node_base(fields[i], with_gradient=False)
        for i in inputs
    ]
    referenced = set()
    for (tt, ac, w, fl) in raw:
        referenced.update(f for (f, p, a) in fl)
        if ac != -1 and ac != tt:
            referenced.add(ac)
    referenced = {i for i in referenced if i in inputs}

    # compiled adjoint stencils (one per stage), cached on the stencil object
    adj = getattr(stencil, "_node_adj", None)
    if adj is None:
        entries, computed, _outs = adjoint_code(raw, n, outputs=(t,), ndim=ndim, temps=tuple(temps))
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
            n_adj_fields = n_comp + 1 + n
            K.data_access_hints(written, [i for i in range(n_adj_fields) if i not in written], [])
            compiled.append(K)
        adj = (computed, compiled)
        stencil._node_adj = adj
    computed, compiled = adj
    n_comp = len(computed)

    def _fwd():
        scratch = g.lattice(grid, otype_t)
        full = [None] * n
        full[t] = scratch
        for k, i in enumerate(inputs):
            full[i] = value_of(children[k])
            if is_node(full[i]):
                raise NotImplementedError(
                    "stencil node mode: nested (non-plain) values are not supported yet")
        stencil(*full)
        return scratch

    def _adj_flow(z, ci):
        cached = getattr(z, "_stencil_adj", None)
        if cached is None or cached[0] is not z.gradient:
            slots = [g.lattice(grid, otype_t) for _ in range(n_comp)]
            for s in slots:
                s[:] = 0
            vals = [None] * n
            vals[t] = z.value
            for k, i in enumerate(inputs):
                vals[i] = value_of(children[k])
                if is_node(vals[i]):
                    raise NotImplementedError(
                        "stencil node mode: nested (non-plain) values are not supported yet")
            if is_node(z.gradient):
                raise NotImplementedError(
                    "stencil node mode: nested (non-plain) flows are not supported yet")
            for K in compiled:
                K(*(slots + [z.gradient] + vals))
            cached = (z.gradient, slots)
            z._stencil_adj = cached
        return cached[1][computed.index(ci)]

    def _backward(z):
        for c, ci in zip(children, inputs):
            if c.with_gradient and ci in referenced:
                accum(c, _adj_flow(z, ci), 1)

    z = fields[t]
    z._forward = _fwd
    z._children = children
    z._backward = _backward
    z.value = None
    z.gradient = None
    z._tag = f"stencil({len(points)} points, {len(stencil.code)} lines of code)"
    return z
