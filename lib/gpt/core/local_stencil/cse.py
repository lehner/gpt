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
# Common-subexpression elimination for the code of a local matrix stencil:
# the execution plan of a kernel, not its meaning (the stencil keeps its
# original code, which the AD derivation reads).
#
# Greedy pair replacement (Re-Pair): the most frequent adjacent factor pair
# is replaced by a per-site temporary (a kernel-owned field read at the zero
# point), until no pair occurs min_uses times.  A pair and its adjoint
# reversal (b^dag a^dag) share one temporary, read with the adjoint flag.
# Temporaries may be built from temporaries.
#
# A factor is a site function of the fields at the time of its read: the
# kernel reads zero-point factors live, in code order, so a field the kernel
# writes is identified by its write version (the number of earlier writes)
# and is eligible only at the zero point (a shifted read of a written field
# is not a per-site value); fields the kernel never writes are eligible at
# any point.  Each temporary is defined just before its first use, where all
# of its factors have the versions of that use.
#
# Cost: a temporary is written once and read per use from a per-block
# buffer, which costs about as much as the products it saves (the kernels are
# dominated by factor fetches, not arithmetic).  Measured on the AD adjoint
# kernels (aarch64): kernels whose products drop by >= 18% run 4-14% faster,
# those at <= 12% are neutral or up to 7% slower.  So a combined plan is
# used only if it saves at least min_saving of the products.
#
# Results change only by the association of products (rounding).
import collections

min_saving_default = 0.15


def cse(points, code, temporaries=(), min_uses=2, min_saving=None):
    # points: list of point tuples; code: [(target, accumulate, weight,
    # [(field, point_index, adj), ...]), ...] (or the parsed dicts).
    # Returns (points, code, temporaries) of the executed kernel; the new
    # temporaries have indices above every field index of the code, so the
    # positions of the passed fields do not change.  Returns None if nothing
    # is combined, or if fewer than a fraction min_saving of the products are
    # saved.
    code = [
        (c["target"], c["accumulate"], c["weight"], list(c["factor"])) if isinstance(c, dict) else c
        for c in code
    ]
    ndim = len(points[0])
    zero = (0,) * ndim
    points = [tuple(p) for p in points]

    written = collections.Counter()
    base = max([t for t in temporaries] + [max(t, a) for (t, a, w, fl) in code]
               + [f for (t, a, w, fl) in code for (f, p, aa) in fl]) + 1

    # symbols: (kind, index, version, point, adj); kind "f" = field factor,
    # "t" = new temporary (version 0, zero point), "x" = ineligible factor (a
    # barrier; index = the original factor)
    seqs = []
    for (t, a, w, fl) in code:
        s = []
        for (f, p, aa) in fl:
            pt = points[p]
            if f in written and pt != zero:
                s.append(("x", (f, p, aa), 0, zero, 0))
            else:
                s.append(("f", f, written[f], pt, aa))
        seqs.append(s)
        written[t] += 1

    def adj(x):
        return x[:4] + (1 - x[4],)

    def canon(a, b):
        q = (adj(b), adj(a))
        return ((a, b), 0) if (a, b) <= q else (q, 1)

    def pairs(s):
        for i in range(len(s) - 1):
            if s[i][0] != "x" and s[i + 1][0] != "x":
                yield i, canon(s[i], s[i + 1])

    defs = []
    while True:
        cnt = collections.Counter()
        for s in seqs:
            last = {}
            for i, (c, _) in pairs(s):
                # non-overlapping occurrences (a run a a a holds one a*a)
                if last.get(c, -2) == i - 1:
                    continue
                last[c] = i
                cnt[c] += 1
        if not cnt:
            break
        best, n = cnt.most_common(1)[0]
        if n < min_uses:
            break
        k = len(defs)
        defs.append(best)
        for j, s in enumerate(seqs):
            out, i = [], 0
            while i < len(s):
                if i < len(s) - 1 and s[i][0] != "x" and s[i + 1][0] != "x":
                    c, flip = canon(s[i], s[i + 1])
                    if c == best:
                        out.append(("t", k, 0, zero, flip))
                        i += 2
                        continue
                out.append(s[i])
                i += 1
            seqs[j] = out

    if not defs:
        return None
    if min_saving is None:
        min_saving = min_saving_default
    before = sum(max(len(fl) - 1, 0) for (t, a, w, fl) in code)
    after = len(defs) + sum(max(len(x) - 1, 0) for x in seqs)
    if before - after < min_saving * before:
        return None

    points = list(points)
    if zero not in points:
        points.append(zero)
    pidx = {p: i for i, p in enumerate(points)}

    def factor(sym):
        if sym[0] == "x":
            return sym[1]
        if sym[0] == "t":
            return (base + sym[1], pidx[zero], sym[4])
        return (sym[1], pidx[sym[3]], sym[4])

    new_code, emitted = [], set()

    def emit_def(k):
        # temporaries are defined just before their first use (depth first)
        if k in emitted:
            return
        for sym in defs[k]:
            if sym[0] == "t":
                emit_def(sym[1])
        emitted.add(k)
        new_code.append((base + k, -1, 1.0, [factor(sym) for sym in defs[k]]))

    for (t, a, w, fl), s in zip(code, seqs):
        for sym in s:
            if sym[0] == "t":
                emit_def(sym[1])
        new_code.append((t, a, w, [factor(sym) for sym in s]))

    return points, new_code, tuple(sorted(temporaries)) + tuple(base + k for k in range(len(defs)))
