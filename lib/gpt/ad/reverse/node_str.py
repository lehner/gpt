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
# str(node): the graph of a node as a listing of temporaries, for reading
# off its cost.
#
#   x0 : lattice(su_n(3))                                  the leaves
#   t0 = x0 * x1          #  [1] lattice(su_n(3))  used 2x
#   t1 = x0 * x1          #  [1] lattice(su_n(3))  RECOMPUTES t0
#   y = inner_product(t0, t0) + t1   # [2] complex
#   -- summary: op counts per op and type, peak live lattice values
#
# Every computed node is evaluated once per pass.  A node used once is
# inlined into its consumer; a node gets its own line (a temporary) if it is
# shared, the root, structurally identical to another node (computed twice,
# marked RECOMPUTES), or if an expression grows longer than the width (then
# its longest inlined operand is hoisted).  [k] counts the lattice ops of a
# line (ops with a lattice result or operand), the cost unit of the graph.
#
import numbers
import numpy as np
import gpt as g

# binary primitives printed infix, with their precedence
_infix = {"+": 1, "-": 1, "*": 2, "/": 2}
_atom = 9
# static arguments that are implementation details
_hidden_statics = {"n_block", "use_accelerator", "c", "reductions", "lattice", "matrix", "branch", "t"}


def _type(c, grids):
    t = c.tag
    if t[0] is list:
        return f"list[{t[2]}]({_type(t[1], grids)})"
    if t[0] is np.ndarray:
        return f"array{tuple(t[1])}"
    if t[0] is complex:
        return "complex"
    ot = t[-1].__name__.replace("ot_", "").replace("matrix_su_n_fundamental_group", "su_n")
    if t[0] is g.lattice:
        # (the grid is named only if the graph has more than one)
        grid = grids.setdefault(id(t[1].obj), f"grid{len(grids)}")
        return f"lattice({ot})" + ("" if grid == "grid0" else f"@{grid}")
    return f"tensor({ot})"


def _is_lattice(c):
    t = c.tag
    return t[0] is g.lattice or (t[0] is list and t[1].tag[0] is g.lattice)


def _lattice_op(n):
    return _is_lattice(n._container) or any(_is_lattice(c._container) for c in n._children)


def _number(v):
    if isinstance(v, complex) and v.imag == 0:
        v = v.real
    if isinstance(v, float) and v == int(v):
        return str(int(v)) + "."
    return str(v)


def _constant_number(n):
    return n._forward is None and not n.with_gradient and isinstance(n.value, numbers.Number)


def _simple(v):
    if isinstance(v, bool):
        return False
    if isinstance(v, (numbers.Number, str)):
        return True
    return isinstance(v, tuple) and all(isinstance(e, numbers.Number) for e in v)


def _statics(n):
    return getattr(n, "_static", {})


def _structure_keys(nodes):
    # per node a key that is equal for structurally identical subgraphs: the
    # same op with the same statics, type and children (leaves by identity,
    # constant numbers by value)
    key = {}
    for n in nodes:
        if n._forward is None:
            key[n] = ("number", _number(n.value)) if _constant_number(n) else ("leaf", id(n))
            continue
        st = []
        for k, v in sorted(_statics(n).items()):
            if _simple(v) or isinstance(v, bool):
                st.append((k, repr(v)))
            elif hasattr(v, "tag"):
                # (a container)
                st.append((k, str(v)))
            else:
                st.append((k, id(v)))
        key[n] = (n._tag, tuple(st), str(n._container), tuple(key[c] for c in n._children))
    return key


class _listing:
    def __init__(self, root, width):
        from gpt.ad.reverse.node import traverse, needed_values

        self.root = root
        self.width = width
        self.nodes = []
        self.free = traverse(self.nodes, root)
        self.needed_values = needed_values
        self.uses = {}
        for n in self.nodes:
            for c in n._children:
                self.uses[c] = self.uses.get(c, 0) + 1
        self.grids = {}
        self.names = {}
        self.counter = {}
        self.lines = []
        # per node: (text, precedence, lattice ops, inlined)
        self.text = {}

        key = _structure_keys(self.nodes)
        first = {}
        self.recomputes = {}
        for n in self.nodes:
            if n._forward is not None:
                if key[n] in first:
                    self.recomputes[n] = first[key[n]]
                else:
                    first[key[n]] = n
        self.recomputed = set(self.recomputes.values())

    def leaf_names(self):
        # user names (rad.node(x, name=...)) and x0, x1, ... for the others
        taken = set(n._name for n in self.nodes if n._forward is None and n._name is not None)
        count = {}
        out = []
        for n in self.nodes:
            if n._forward is not None or _constant_number(n):
                continue
            if n._name is not None:
                count[n._name] = count.get(n._name, 0) + 1
                name = n._name if count[n._name] == 1 else f"{n._name}#{count[n._name]}"
            else:
                name = self.fresh("x", taken)
            self.names[n] = name
            const = "" if n.with_gradient else "   (constant)"
            out.append(f"{name} : {_type(n._container, self.grids)}{const}")
        self.taken = taken | set(self.names.values())
        return out

    def fresh(self, prefix, taken):
        # prefix0, prefix1, ... (skipping the names taken)
        i = self.counter.get(prefix, 0)
        while f"{prefix}{i}" in taken:
            i += 1
        self.counter[prefix] = i + 1
        taken.add(f"{prefix}{i}")
        return f"{prefix}{i}"

    def operand(self, c, p, right=False):
        s, q = self.text[c][:2]
        # (a right operand of equal precedence is parenthesized: the
        # evaluation order is kept, products of matrices do not commute)
        return "(" + s + ")" if q < p or (right and q == p) else s

    def compose(self, n):
        tag, ch, st = n._tag, n._children, _statics(n)
        if tag is None:
            # (a lazy zero, see node.zero)
            return "zero", _atom
        if tag in _infix and len(ch) == 2:
            p = _infix[tag]
            return f"{self.operand(ch[0], p)} {tag} {self.operand(ch[1], p, True)}", p
        if tag == "**":
            return f"{self.operand(ch[0], _atom)}**{st['n']}", 3
        if tag in ("list_element", "element") and len(ch) == 1:
            return f"{self.operand(ch[0], _atom)}[{st['index']}]", _atom
        args = [self.text[c][0] for c in ch]
        if tag == "cshift":
            args += [str(st["direction"]), "%+d" % st["displacement"]]
        else:
            args += [
                f"{k}={v}" for k, v in st.items() if k not in _hidden_statics and _simple(v)
            ]
        return f"{tag}({', '.join(args)})", _atom

    def emit(self, n, s, ops):
        if n is self.root:
            self.names[n] = "y" if "y" not in self.taken else self.fresh("y", self.taken)
        else:
            self.names[n] = self.fresh("t", self.taken)
        note = f"  used {self.uses[n]}x" if self.uses.get(n, 0) > 1 else ""
        if n in self.recomputes:
            note += f"  RECOMPUTES {self.names[self.recomputes[n]]}"
        self.lines.append((self.names[n], s, _type(n._container, self.grids), ops, note))
        self.text[n] = (self.names[n], _atom, 0, False)

    def body(self):
        for n in self.nodes:
            if n._forward is None:
                s = _number(n.value) if _constant_number(n) else self.names[n]
                self.text[n] = (s, _atom, 0, False)
                continue
            while True:
                s, p = self.compose(n)
                inlined = [c for c in n._children if self.text[c][3]]
                if len(s) <= self.width or not inlined:
                    break
                c = max(inlined, key=lambda c: len(self.text[c][0]))
                self.emit(c, self.text[c][0], self.text[c][2])
            ops = sum(self.text[c][2] for c in n._children) + (1 if _lattice_op(n) else 0)
            if (
                n is self.root
                or self.uses.get(n, 0) > 1
                or n in self.recomputes
                or n in self.recomputed
            ):
                self.emit(n, s, ops)
            else:
                self.text[n] = (s, p, ops, True)

        if not self.lines:
            return []
        w = max(len(f"{a} = {b}") for a, b, *_ in self.lines)
        wt = max(len(t) for _, _, t, _, _ in self.lines)
        out = []
        for a, b, t, ops, note in self.lines:
            cost = f"[{ops}]" if ops else ""
            out.append(f"{a} = {b}".ljust(w) + f"   # {cost:>4} {t.ljust(wt)}{note}".rstrip())
        return out

    def summary(self):
        computed = [n for n in self.nodes if n._forward is not None]
        lattice = [n for n in computed if _lattice_op(n)]
        out = [
            f"-- {len(computed)} ops, {len(lattice)} lattice ops [..], "
            f"recomputed subexpressions: {len(self.recomputes)}"
        ]
        count = {}
        for n in lattice:
            k = f"{n._tag or 'zero'} -> {_type(n._container, self.grids)}"
            count[k] = count.get(k, 0) + 1
        for k in sorted(count, key=lambda k: -count[k]):
            out.append(f"--   {count[k]:4d} x {k}")
        # the lattice values alive at once in a forward-only pass (values
        # freed after their last use), and the ones a reverse pass keeps for
        # the backward (see needed_values)
        live, peak = set(), 0
        for n in computed:
            live.add(n)
            peak = max(peak, sum(1 for m in live if _is_lattice(m._container)))
            live -= set(self.free[n])
        kept = [
            n
            for n in self.needed_values(self.nodes)
            if n._forward is not None and _is_lattice(n._container)
        ]
        out.append(
            f"-- peak live lattice values: forward only {peak}, kept for the backward {len(kept)}"
        )
        return out


def node_str(root, width=60):
    if root._forward is None:
        return f"{root._name or 'leaf'} : {_type(root._container, {})}"
    listing = _listing(root, width)
    return "\n".join(listing.leaf_names() + listing.body() + listing.summary())
