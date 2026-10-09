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
# Drawing of the graph of a g.ml.pack (matplotlib, imported only when drawing)
#
import re


def draw(graph, ax=None, inputs=None):
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

    calls, inputs = graph._inputs(inputs)
    ink, ink2, muted, frame, surface = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"
    palette = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (all-pairs safe)

    def source(s):
        return ("in", s.id) if s.call is None else ("call", s.call.id)

    # edges: (source node, output port, #outputs, target node, input port,
    # #ports, source label, target label, style)
    edges = []
    for c in calls:
        f = c.function
        ports = [(slot, s, None, "solid") for slot, s in zip(f.input_names(), c.inputs)]
        ports += [
            (f._parameters.names[j], s, k, "dashed")
            for j, (s, k) in sorted(c.connections.items())
        ]
        for i, (slot, s, k, style) in enumerate(ports):
            edges.append(
                _draw_edge(
                    source(s),
                    s,
                    ("call", c.id),
                    i,
                    len(ports),
                    slot if len(ports) > 1 or style == "dashed" else None,
                    k,
                    style,
                )
            )
    for name, s in graph.outputs.items():
        edges.append(_draw_edge(source(s), s, ("out", name), 0, 1, None, None, "solid"))

    # columns (longest path) and waypoints of edges across several columns
    column = {("in", s.id): 0 for s in inputs}
    for c in calls:
        column[("call", c.id)] = 1 + max(
            [column[e["a"]] for e in edges if e["b"] == ("call", c.id)], default=0
        )
    last = 1 + max(column.values(), default=0)
    for name in graph.outputs:
        column[("out", name)] = last
    height = {k: (0.6 if k[0] == "call" else 0.4) for k in column}
    # widths from the text (call name bold 10pt, class 8pt; ~0.085 per character)
    width = {("in", s.id): max(1.0, 0.09 * len(s.label()) + 0.4) for s in inputs}
    width.update({("out", n): max(1.0, 0.09 * len(n) + 0.4) for n in graph.outputs})
    for c in calls:
        if c.label is not None:
            width[("call", c.id)] = max(1.0, 0.11 * _visible_length(c.label) + 0.5)
        else:
            width[("call", c.id)] = max(
                1.8, 0.09 * len(c.name) + 0.4, 0.072 * len(type(c.function).__name__) + 0.4
            )
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
        ax.add_patch(
            FancyBboxPatch(
                (x[k] - w / 2, y[k] - h / 2),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={rounding}",
                facecolor=surface,
                edgecolor=edge,
                linewidth=lw,
                zorder=2,
            )
        )
        for dy, t, kw in text:
            ax.text(x[k], y[k] + dy, t, ha="center", va="center", zorder=3, **kw)

    for s in inputs:
        box(("in", s.id), [(0, s.label(), dict(color=ink, fontsize=10))], frame, 1.5, 0.2)
    for name in graph.outputs:
        box(("out", name), [(0, name, dict(color=ink, fontsize=10))], frame, 1.5, 0.2)
    for c in calls:
        fid = id(c.function)
        color = (
            palette[shared.index(fid)]
            if fid in shared and shared.index(fid) < len(palette)
            else frame
        )
        if c.label is not None:
            text = [(0.0, c.label, dict(color=ink, fontsize=12))]
        else:
            text = [
                (0.11, c.name, dict(color=ink, fontsize=10, fontweight="bold")),
                (-0.13, type(c.function).__name__, dict(color=ink2, fontsize=8)),
            ]
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
        ax.add_patch(
            FancyArrowPatch(
                path=Path(verts, codes),
                arrowstyle="-|>",
                mutation_scale=10,
                color=ink2,
                linewidth=1.0,
                linestyle=e["style"],
                zorder=1,
            )
        )
        if e["slot"]:
            ax.text(
                pts[-1][0] - 0.08,
                pts[-1][1] + 0.09,
                e["slot"],
                ha="right",
                va="center",
                color=ink2,
                fontsize=7,
                zorder=3,
            )
        if e["output"] and (e["a"], e["out"]) not in labeled:
            labeled.add((e["a"], e["out"]))
            ax.text(
                pts[0][0] + 0.08,
                pts[0][1] + 0.09,
                e["output"],
                ha="left",
                va="center",
                color=ink2,
                fontsize=7,
                zorder=3,
            )

    ax.set_xlim(-col_width[0] / 2 - 0.3, col_x[-1] + col_width[-1] / 2 + 0.3)
    ax.set_ylim(min(y.values()) - 0.6, max(y.values()) + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(surface)
    fig.patch.set_facecolor(surface)
    return fig


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
    return dict(
        a=a,
        out=out,
        n_out=n_out,
        b=b,
        port=port,
        n_ports=n_ports,
        slot=slot,
        output=output,
        style=style,
    )
