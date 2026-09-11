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
#     (gpt.core.local_stencil.adjoint), one kernel per level.
#
# Current limitations (plain level only):
#   - a single target field, which must be a node
#   - the first write of the target must be fresh (accumulate = -1), so the
#     target's old value is not part of the computation
#   - the target's old value must be plain (no nested nodes)
#   - node values/flows must be plain when the kernels run (a 1-deep graph
#     reversed with plain flow); nested (multi-deep) use raises
#
import gpt as g
from gpt.ad.reverse.util import value_of, is_node, nodify, accum


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
    # Levels: a compiled stencil snapshots all read fields before executing
    # its entries, so an entry reading a running slot must run in a later
    # stencil than the entries that write that slot.  Entries are emitted in
    # reverse forward order and assigned to levels: an entry reading no
    # slots is level 0; otherwise level = 1 + max(level of the already
    # emitted entries writing the slots it reads).  The caller compiles one
    # stencil per level and runs them in level order with persistent slot
    # lattices.  (Codes without temps are single-level.)
    #
    # returns (entries, computed, outs) where
    #   entries  : list of (level, (target_slot, accumulate, weight, flist))
    #   computed : slot order = [inputs..., "v:<temp>:<version>"...]; the
    #              adjoint field layout is [slot_i for i in computed]
    #              + [psi_t for t in outs] + [f_i for i in range(n_fields)]
    #   levels are per entry as documented above; the caller runs one
    #              compiled stencil per level, in level order, with
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

    def version_readers(s, k):
        # forward entries reading version (s, k)
        w_k = writers[s][k]
        w_next = writers[s][k + 1] if k + 1 < len(writers[s]) else None
        rs = [
            i for i in range(w_k + 1, len(code) if w_next is None else w_next)
            if code[i][1] == s and code[i][1] != code[i][0]
        ]
        if w_next is not None and code[w_next][1] == s:
            rs.append(w_next)  # self-acc rewrite reads the previous version
        return rs

    # levels: the entries of a version's writer read the version's flow
    # slot, which is completed by all of the version's readers
    L = [0] * len(code)
    changed = True
    while changed:
        changed = False
        for s in tmps:
            for k in range(len(writers[s])):
                need = 1 + max([L[r] for r in version_readers(s, k)], default=-1)
                e = writers[s][k]
                if L[e] < need:
                    L[e] = need
                    changed = True

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

    def emit(level, target, weight, flist):
        # every entry accumulates into its slot; the caller
        # zero-initializes the slot lattices
        entries.append([level, (target, target, weight, flist)])
        return len(entries) - 1

    for i in range(len(code) - 1, -1, -1):
        target, acc, weight, factors = code[i]
        k = len(factors)
        level = L[i]
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
            emit(level, slot[("in", i_m)], w, flist)
        if acc != -1 and acc != target:
            if acc in inputs:
                emit(level, slot[("in", acc)], 1.0, [(phi[0], zero, 0)])
            elif acc in tmps:
                # read the version of the last writer before this entry
                ks = [kk for kk in range(len(writers[acc])) if writers[acc][kk] < i]
                assert ks, "acc reads temp %d before it is written" % acc
                emit(level, slot[("v", acc, ks[-1])], 1.0, [(phi[0], zero, 0)])
        if target in tmps and acc == target:
            kv = writers[target].index(i)
            if kv > 0:
                # self-acc handoff: flow of the previous version receives
                # the flow of this version
                emit(level, slot[("v", target, kv - 1)], 1.0, [(phi[0], zero, 0)])
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
    assert len(targets) == 1, "stencil node mode currently supports a single target"
    t = targets[0]
    assert is_node(fields[t]), "stencil target must be a node"
    first = {}
    for (tt, ac, w, fl) in raw:
        first.setdefault(tt, ac)
    assert first[t] == -1, (
        "stencil node mode: first write of the target must be fresh (acc=-1)")

    tv = value_of(fields[t])
    assert not is_node(tv), "stencil node mode: target's old value must be plain"
    otype_t = tv.otype

    inputs = [i for i in range(n) if i != t]
    children = [nodify(fields[i]) for i in inputs]
    referenced = set()
    for (tt, ac, w, fl) in raw:
        referenced.update(f for (f, p, a) in fl)
        if ac != -1 and ac != tt:
            referenced.add(ac)
    referenced = {i for i in referenced if i in inputs}

    # compiled adjoint stencils (one per level), cached on the stencil object
    adj = getattr(stencil, "_node_adj", None)
    if adj is None:
        entries, computed, _outs = adjoint_code(raw, n, outputs=(t,), ndim=ndim)
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
