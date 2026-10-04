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
#
# Composite functions from symbolic calls:
#
#   x1, x2 = g.ml.symbols("x1", "x2")
#   a, = f1([x1], name="enc")
#   b, z = f2([x1, x2])                       # call named "f2" (class name)
#   y, = f3([a, b], parameters={"w": z})      # f3's parameter w computed by f2
#   net = g.ml.pack(y=y, z=z).function()      # inputs [x1, x2]
#
# A function called on symbols records the call and returns symbols for its
# outputs.  pack names the outputs; function() collects the calls the outputs
# depend on and returns a g.ml.composite.
#
import itertools
import re
import gpt as g
from gpt.ml.function import function, _check_name, _check_names, _named_storage, _storage_list

_creation = itertools.count()


class _symbol:
    # a free symbol (an input; call is None) or output index of a call
    def __init__(self, name=None, call=None, index=None):
        self.name = name
        self.call = call
        self.index = index
        self.id = next(_creation)

    def __repr__(self):
        return f"symbol({self.label()})"

    def label(self):
        # the input's name, or call.output
        if self.call is None:
            return self.name
        return f"{self.call.name}.{self.call.function.output_names()[self.index]}"

    def describe(self):
        return pack._of([self]).describe()

    def draw(self, ax=None):
        return pack._of([self]).draw(ax)


class _call:
    def __init__(self, name, function, inputs, connections, label=None):
        self.name = name
        self.label = label
        self.function = function
        self.inputs = inputs
        # flat parameter index of the function -> (symbol, list element or None)
        self.connections = connections
        self.id = next(_creation)
        self.outputs = [_symbol(call=self, index=i) for i in range(len(function.output_names()))]


def symbols(*names):
    _check_names(list(names))
    return [_symbol(name=n) for n in names]


def is_symbolic(inputs, parameters):
    return isinstance(parameters, dict) or any(isinstance(x, _symbol) for x in inputs)


def record_call(f, inputs, parameters, name, label=None):
    name = type(f).__name__ if name is None else name
    _check_name(name)
    parameters = {} if parameters is None else parameters
    if not isinstance(parameters, dict):
        raise TypeError(f"{name}: parameters of a symbolic call are a dict {{slot: symbol}}")
    if len(inputs) != len(f.input_names()):
        raise ValueError(f"{name}: expected {len(f.input_names())} inputs, got {len(inputs)}")
    for s in list(inputs) + list(parameters.values()):
        if not isinstance(s, _symbol):
            raise TypeError(f"{name}: symbolic calls take symbols only, got {type(s)}")
    connections = {}
    for slot, s in parameters.items():
        r = f._parameters.find(slot)
        if r is None:
            raise KeyError(f"{name}: no parameter {slot!r}")
        offset, n = r
        for j, k in [(offset, None)] if n is None else [(offset + k, k) for k in range(n)]:
            if j in connections:
                raise ValueError(f"{name}: parameter {slot!r} connected twice")
            connections[j] = (s, k)
    return _call(name, f, list(inputs), connections, label).outputs


def _subgraph(roots):
    # the calls the roots depend on (in order of creation, which is a
    # dependency order) and the free symbols they use (in order of creation)
    calls, free = {}, {}
    stack = list(roots)
    while stack:
        s = stack.pop()
        if s.call is None:
            free[s.id] = s
        elif s.call.id not in calls:
            calls[s.call.id] = s.call
            stack += s.call.inputs + [t for t, _ in s.call.connections.values()]
    return [calls[k] for k in sorted(calls)], [free[k] for k in sorted(free)]


class _layout:
    # the parameters and constants of a set of calls: each function once, in
    # the order of its first call, whose name prefixes its slots; a parameter
    # slot is owned if at least one call of its function leaves it
    # unconnected.  index[id(f)][j] is the position of slot j of f, prefix[id(f)]
    # the name of its first call.
    def __init__(self, calls):
        self.prefix = {}
        for c in calls:
            self.prefix.setdefault(id(c.function), c.name)
        self.functions = list({id(c.function): c.function for c in calls}.values())
        self.names, self.entries, self.index = [], [], {}
        self.constant_names, self.constant_entries = [], []
        for f in self.functions:
            prefix = self.prefix[id(f)]
            f_calls = [d for d in calls if d.function is f]
            owned = [
                j
                for j in range(len(f._parameters.values))
                if any(j not in d.connections for d in f_calls)
            ]
            self.index[id(f)] = {j: len(self.entries) + k for k, j in enumerate(owned)}
            self.names += [f"{prefix}.{f._parameters.names[j]}" for j in owned]
            self.entries += [_entry(f._parameters.values, j) for j in owned]
            self.constant_names += [f"{prefix}.{name}" for name in f._constants.names]
            self.constant_entries += [_entry(f._constants.values, j) for j in range(len(f._constants.values))]


class pack:
    """Named outputs (pack(y=a, z=b)) of a function under construction: the
    graph of the calls they depend on.  function() turns it into a
    g.ml.composite; describe() and draw() show it."""

    def __init__(self, **outputs):
        _check_names(list(outputs.keys()))
        for name, s in outputs.items():
            if not isinstance(s, _symbol):
                raise TypeError(
                    f"Output {name!r} must be a symbol, got {type(s)} (unpack calls: a, = f([x]))"
                )
        self.outputs = outputs

    @classmethod
    def _of(cls, roots):
        # an unnamed pack (for inspecting symbols), labeled by the symbols
        r = cls()
        r.outputs = {s.label(): s for s in roots}
        return r

    def _inputs(self, inputs):
        # inputs: the free symbols in the order of the function's inputs;
        # default: those the outputs depend on, in order of creation
        calls, free = _subgraph(self.outputs.values())
        if inputs is None:
            return calls, free
        for s in inputs:
            if not isinstance(s, _symbol) or s.call is not None:
                raise TypeError(f"Inputs must be free symbols, got {s!r}")
            if not any(s is t for t in free):
                raise ValueError(f"Input {s!r} is not used by the outputs")
        unbound = [s for s in free if not any(s is t for t in inputs)]
        if unbound:
            raise ValueError(f"Symbols {unbound} are used but not inputs")
        return calls, list(inputs)

    def function(self, inputs=None):
        calls, inputs = self._inputs(inputs)
        names = [c.name for c in calls]
        duplicates = sorted(set(n for n in names if names.count(n) > 1))
        if duplicates:
            raise ValueError(f"Duplicate call names {duplicates}: name the calls (name=...)")
        return composite(self, inputs, calls)

    def describe(self, inputs=None):
        """A text listing of the graph: the inputs, one line per call (its
        arguments, connected parameters after ';', and the parameters and
        constants it takes from storage, by their names in the composite),
        and the outputs."""
        calls, inputs = self._inputs(inputs)
        layout = _layout(calls)
        rows = []
        for c in calls:
            f = c.function
            args = [f"{n}={s.label()}" for n, s in zip(f.input_names(), c.inputs)]
            connected = [
                f"{f._parameters.names[j]}={s.label() if k is None else s.label() + '[' + str(k) + ']'}"
                for j, (s, k) in sorted(c.connections.items())
            ]
            call = f"{type(f).__name__}({', '.join(args)}{'; ' + ', '.join(connected) if connected else ''})"
            prefix = layout.prefix[id(f)]
            stored = [f"{prefix}.{n}" for j, n in enumerate(f._parameters.names) if j not in c.connections]
            constants = [f"{prefix}.{n}" for n in f._constants.names]
            rows.append((c.name, call, stored, constants))
        w0 = max([len(r[0]) for r in rows], default=0)
        w1 = max([len(r[1]) for r in rows], default=0)
        lines = [f"inputs: {', '.join(s.label() for s in inputs)}"]
        for name, call, stored, constants in rows:
            line = f"  {name.ljust(w0)} = {call.ljust(w1)}"
            if stored:
                line += f"  parameters: {', '.join(stored)}"
            if constants:
                line += f"  constants: {', '.join(constants)}"
            lines.append(line.rstrip())
        lines.append(
            f"outputs: {', '.join(n if n == s.label() else f'{n}={s.label()}' for n, s in self.outputs.items())}"
        )
        return "\n".join(lines)

    def draw(self, ax=None, inputs=None):
        """Draw the graph with matplotlib (imported here only): inputs left,
        outputs right, a box per call (name, function class; nested
        composites as single boxes) in the column after its latest source.
        Edges leave a box at the port of their output and enter at the port
        of their slot (inputs, then connected parameters: dashed), labeled
        with the slot (and the output, if a call has several); edges across
        several columns pass through free lanes.  The calls of a shared
        function have a common border color (first three shared functions)
        and name the call they share with.  Draws into ax if given; returns
        the figure."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
        from matplotlib.path import Path

        calls, inputs = self._inputs(inputs)
        ink, ink2, muted, frame, surface = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"
        palette = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (all-pairs safe)

        def source(s):
            return ("in", s.id) if s.call is None else ("call", s.call.id)

        # edges: (source node, output port, #outputs, target node, input port,
        # #ports, source label, target label, style)
        edges = []
        for c in calls:
            f = c.function
            n_out = len(f.output_names())
            ports = [(slot, s, None, "solid") for slot, s in zip(f.input_names(), c.inputs)]
            ports += [(f._parameters.names[j], s, k, "dashed") for j, (s, k) in sorted(c.connections.items())]
            for i, (slot, s, k, style) in enumerate(ports):
                edges.append(_draw_edge(source(s), s, ("call", c.id), i, len(ports), slot if len(ports) > 1 or style == "dashed" else None, k, style))
        for name, s in self.outputs.items():
            edges.append(_draw_edge(source(s), s, ("out", name), 0, 1, None, None, "solid"))

        # columns (longest path) and waypoints of edges across several columns
        column = {("in", s.id): 0 for s in inputs}
        for c in calls:
            column[("call", c.id)] = 1 + max([column[e["a"]] for e in edges if e["b"] == ("call", c.id)], default=0)
        last = 1 + max(column.values(), default=0)
        for name in self.outputs:
            column[("out", name)] = last
        height = {k: (0.6 if k[0] == "call" else 0.4) for k in column}
        # widths from the text (call name bold 10pt, class 8pt; ~0.085 per character)
        width = {("in", s.id): max(1.0, 0.09 * len(s.label()) + 0.4) for s in inputs}
        width.update({("out", n): max(1.0, 0.09 * len(n) + 0.4) for n in self.outputs})
        for c in calls:
            if c.label is not None:
                width[("call", c.id)] = max(1.0, 0.11 * _visible_length(c.label) + 0.5)
            else:
                width[("call", c.id)] = max(1.8, 0.09 * len(c.name) + 0.4, 0.072 * len(type(c.function).__name__) + 0.4)
        for n, e in enumerate(edges):
            e["via"] = []
            for col in range(column[e["a"]] + 1, column[e["b"]]):
                k = ("via", n, col)
                column[k], height[k], width[k] = col, 0.1, 0.0
                e["via"].append(k)

        def port_dy(k, i, n):
            return 0.0 if n == 1 else (0.5 - i / (n - 1)) * 0.7 * height[k]

        # hops (p, q, dy at p, dy at q): the ports only at the ends of an edge
        links = []
        for e in edges:
            nodes = [e["a"]] + e["via"] + [e["b"]]
            for h, (p, q) in enumerate(zip(nodes[:-1], nodes[1:])):
                dp = port_dy(p, e["out"], e["n_out"]) if h == 0 else 0.0
                dq = port_dy(q, e["port"], e["n_ports"]) if h == len(nodes) - 2 else 0.0
                links.append((p, q, dp, dq))

        # rows: sweeps ordering each inner column by the mean row (at the
        # ports) of its neighbors; inputs and outputs keep their order
        y = {}
        cols = [[k for k in column if column[k] == col] for col in range(last + 1)]

        def place(nodes):
            total = sum(height[k] for k in nodes) + 0.45 * (len(nodes) - 1)
            top = total / 2
            for k in nodes:
                y[k] = top - height[k] / 2
                top -= height[k] + 0.45

        def weight(k, forward):
            # the mean row of the predecessors (forward) or successors, at the ports
            if forward:
                ys = [y[p] + dp - dq for p, q, dp, dq in links if q == k and p in y]
            else:
                ys = [y[q] + dq - dp for p, q, dp, dq in links if p == k and q in y]
            return sum(ys) / len(ys) if ys else y.get(k, 0.0)

        for col in range(last + 1):
            place(cols[col])
        for forward in [True, False] * 3:
            for col in range(1, last) if forward else range(last - 1, 0, -1):
                cols[col].sort(key=lambda k: weight(k, forward), reverse=True)
                place(cols[col])

        # columns: the widest box of each plus a gap
        col_width = [max(width[k] for k in cols[col]) for col in range(last + 1)]
        col_x = [0.0]
        for col in range(1, last + 1):
            col_x.append(col_x[-1] + (col_width[col - 1] + col_width[col]) / 2 + 1.2)
        x = {k: col_x[column[k]] for k in column}

        first, shared = {}, []
        for c in calls:
            if id(c.function) in first and id(c.function) not in shared:
                shared.append(id(c.function))
            first.setdefault(id(c.function), c)

        span = max(y.values()) - min(y.values()) if y else 0
        if ax is None:
            fig, ax = plt.subplots(figsize=(col_x[-1] + col_width[-1] + 1.0, 1.0 * span + 1.6))
        fig = ax.figure

        def box(k, text, edge, lw, rounding):
            w, h = width[k], height[k]
            ax.add_patch(FancyBboxPatch((x[k] - w / 2, y[k] - h / 2), w, h,
                                        boxstyle=f"round,pad=0,rounding_size={rounding}",
                                        facecolor=surface, edgecolor=edge, linewidth=lw, zorder=2))
            for dy, t, kw in text:
                ax.text(x[k], y[k] + dy, t, ha="center", va="center", zorder=3, **kw)

        for s in inputs:
            box(("in", s.id), [(0, s.label(), dict(color=ink, fontsize=10))], frame, 1.5, 0.2)
        for name in self.outputs:
            box(("out", name), [(0, name, dict(color=ink, fontsize=10))], frame, 1.5, 0.2)
        for c in calls:
            fid = id(c.function)
            color = palette[shared.index(fid)] if fid in shared and shared.index(fid) < len(palette) else frame
            if c.label is not None:
                text = [(0.0, c.label, dict(color=ink, fontsize=12))]
            else:
                text = [(0.11, c.name, dict(color=ink, fontsize=10, fontweight="bold")),
                        (-0.13, type(c.function).__name__, dict(color=ink2, fontsize=8))]
            if fid in shared and first[fid] is not c:
                shown = first[fid].label if first[fid].label is not None else first[fid].name
                text.append((-0.43, f"shares {shown}", dict(color=muted, fontsize=7)))
            box(("call", c.id), text, color, 2.0 if fid in shared else 1.5, 0.08)

        def port(k, i, n, side):
            # attachment point i of n on the left (side=-1) or right (+1) of k
            return (x[k] + side * width[k] / 2, y[k] + port_dy(k, i, n))

        labeled = set()
        for e in edges:
            pts = [port(e["a"], e["out"], e["n_out"], 1)] + [(x[k], y[k]) for k in e["via"]]
            pts += [port(e["b"], e["port"], e["n_ports"], -1)]
            verts, codes = [pts[0]], [Path.MOVETO]
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                d = (x1 - x0) / 2
                verts += [(x0 + d, y0), (x1 - d, y1), (x1, y1)]
                codes += [Path.CURVE4] * 3
            ax.add_patch(FancyArrowPatch(path=Path(verts, codes), arrowstyle="-|>", mutation_scale=10,
                                         color=ink2, linewidth=1.0, linestyle=e["style"], zorder=1))
            if e["slot"]:
                ax.text(pts[-1][0] - 0.08, pts[-1][1] + 0.09, e["slot"], ha="right", va="center",
                        color=ink2, fontsize=7, zorder=3)
            if e["output"] and (e["a"], e["out"]) not in labeled:
                labeled.add((e["a"], e["out"]))
                ax.text(pts[0][0] + 0.08, pts[0][1] + 0.09, e["output"], ha="left", va="center",
                        color=ink2, fontsize=7, zorder=3)

        ax.set_xlim(-col_width[0] / 2 - 0.3, col_x[-1] + col_width[-1] / 2 + 0.3)
        ax.set_ylim(min(y.values()) - 0.6, max(y.values()) + 0.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(surface)
        fig.patch.set_facecolor(surface)
        return fig


class composite(function):
    """The function of a pack (see g.ml.pack).  Its parameters are the
    parameter slots of its functions that at least one call leaves
    unconnected, named call.slot with the first call of each function; its
    constants are all constants of its functions.  It holds no values: its
    storage lists point into the storages of its functions, so any number of
    composites share the values of a function.  f["call.name"] addresses the
    function of a call."""

    def __init__(self, graph, inputs, calls):
        super().__init__([s.name for s in inputs], list(graph.outputs.keys()), [])
        self._graph = graph
        self._input_symbols = inputs
        self._output_symbols = list(graph.outputs.values())
        self._calls = calls
        self._functions = {c.name: c.function for c in calls}
        self._layout = layout = _layout(calls)
        self._parameters = _named_storage.shared(layout.names, layout.entries)
        self._constants = _named_storage.shared(layout.constant_names, layout.constant_entries)

    def graph(self):
        return self._graph

    def calls(self):
        return self._calls

    def describe(self):
        return self._graph.describe(self._input_symbols)

    def draw(self, ax=None):
        return self._graph.draw(ax, self._input_symbols)

    def initialize(self, rng, scale=None):
        # each function from its own stream, seeded by the seed of rng and the
        # function's name (its first call), so that its initial values do not
        # depend on the other functions; scale (the distance from the
        # identity, None: the functions' defaults) is passed on if given
        for f in self._layout.functions:
            sub = g.random(f"{rng.seed}/{self._layout.prefix[id(f)]}", rng.engine)
            if scale is None:
                f.initialize(sub)
            else:
                f.initialize(sub, scale=scale)

    def _lookup_function(self, name):
        call, _, rest = name.partition(".")
        if call not in self._functions or rest == "":
            raise KeyError(f"No parameter or constant {name!r}")
        return self._functions[call], rest

    def __getitem__(self, name):
        f, rest = self._lookup_function(name)
        return f[rest]

    def __setitem__(self, name, value):
        f, rest = self._lookup_function(name)
        f[rest] = value

    def _run(self, c, values, parameters):
        # evaluate call c: values maps symbol ids to values, parameters is
        # flat in the order of the composite's slots
        def value(s, k=None):
            v = values[s.id]
            return v if k is None else v[k]

        index = self._layout.index[id(c.function)]
        p = [
            value(*c.connections[j]) if j in c.connections else parameters[index[j]]
            for j in range(len(c.function._parameters.values))
        ]
        for s, y in zip(c.outputs, c.function([value(s) for s in c.inputs], p)):
            values[s.id] = y

    def evaluate(self, inputs, parameters, constants):
        values = {s.id: x for s, x in zip(self._input_symbols, inputs)}
        for c in self._calls:
            self._run(c, values, parameters)
        return [values[s.id] for s in self._output_symbols]

    def calibrate(self, samples):
        # the calls in order: each function is calibrated (at its first call)
        # on the inputs it receives from the samples, then evaluated, so later
        # functions see calibrated earlier ones
        values = [{s.id: x for s, x in zip(self._input_symbols, inputs)} for inputs in samples]
        calibrated = set()
        for c in self._calls:
            if id(c.function) not in calibrated:
                c.function.calibrate([[v[s.id] for s in c.inputs] for v in values])
                calibrated.add(id(c.function))
            for v in values:
                self._run(c, v, self.parameters())


def _visible_length(label):
    # the approximate number of characters a (mathtext) label shows: no $,
    # a command such as \oplus counts as one character
    return len(re.sub(r"\\[a-zA-Z]+", "x", label.replace("$", "")))


def _draw_edge(a, s, b, port, n_ports, slot, k, style):
    # an edge of draw(): from the output port of symbol s (node a) to input
    # port `port` of n_ports of node b; output named if its call has several
    n_out, output = 1, None
    if s.call is not None:
        n_out = len(s.call.function.output_names())
        output = s.call.function.output_names()[s.index] if n_out > 1 else None
    if k is not None:
        output = f"{output or ''}[{k}]"
    out = 0 if s.call is None else s.index
    return dict(a=a, out=out, n_out=n_out, b=b, port=port, n_ports=n_ports, slot=slot, output=output, style=style)


def _entry(values, j):
    # where element j of a storage lives (a composite's entries point through)
    if isinstance(values, _storage_list):
        return values.entries[j]
    return (values, j)
