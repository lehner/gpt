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
# The site-diagonal Jacobian of a node graph as a node:
#
#     J = g.ad.reverse.jacobian(y, x)      J(s) = d y(s) / d x(s)
#
# in the coordinates of the otype of x and y (jacobian_directions,
# jacobian_coordinates, jacobian_otype; SU(N) groups: x -> exp(i e T_a) x,
# the generator coordinates of -i dy y^dag, an ng x ng real matrix per site).
#
# The subgraph between x and y is replayed on k forward tangents (one per
# direction): the values are the nodes of the graph (nothing is recomputed),
# the tangents are new nodes built by the nodes' tangent rules (_jvp: the
# optional jvp of the primitives), batched over the k directions
# where an op has a fused rule (one stencil with k outputs for a zero-point
# stencil, one jet kernel for exp).  J is an ordinary node of the same graph,
# so a function of J (log_det, ...) is differentiated w.r.t. everything by
# ONE ordinary reverse pass -- the second derivatives of the replayed ops come
# from differentiating their tangents.
#
# Site-locality: only site-local ops may lie on a path from x to y (an op
# without a tangent rule raises, e.g. cshift; a stencil reading the varying
# field at a shifted point raises).  A field that reads x at other sites
# (e.g. a staple of the links) must take x from another node: the local map
# gets identity(x) as its x.
#
# chunk = c: J is one primitive of the primal nodes of the subgraph, and the
# tangents are replayed on internal leaves c directions at a time, in the
# forward and again (one internal reverse pass per chunk) in the backward.
# Only one chunk's tangent values are alive at a time, at the cost of the
# replays and of smaller batched kernels (first order).
#
import gpt as g
from gpt.ad.reverse.primitive import primitive
from gpt.ad.reverse.util import container, constant
from gpt.ad.reverse import tangent

# x as a new node of the same value (the boundary of a local map, see above)
identity = primitive(
    "identity",
    lambda x: x,
    lambda c: c.copy(),
    vjp=lambda i, flow, x: flow,
    reads=((),),
    jvp=tangent.linear(lambda t: t),
    fresh=True,
)


def _subgraph(y, x):
    # the nodes on paths from x to y, children first (x first)
    order, depends = [], {}
    stack = [(y, False)]
    while stack:
        n, expanded = stack.pop()
        key = id(n)
        if expanded:
            depends[key] = n is x or any(depends[id(c)] for c in n._children)
            if depends[key]:
                order.append(n)
            continue
        if key in depends:
            continue
        depends[key] = False
        stack.append((n, True))
        for c in n._children:
            if id(c) not in depends:
                stack.append((c, False))
    if not depends[id(y)]:
        raise ValueError("jacobian: y does not depend on x")
    return order


def _tangents(order, x, directions, node_of=None):
    # the tangents of the nodes in order (k directions), replayed with the
    # node_of(n) standing for the value of n (default: n itself)
    node_of = (lambda n: n) if node_of is None else node_of
    tangents = {id(x): directions}
    for n in order:
        if n is x:
            continue
        if n._jvp is None:
            raise NotImplementedError(
                f"jacobian: no tangent rule for the node {n._tag} (not site-local, or no jvp)"
            )
        tangents[id(n)] = n._jvp(
            node_of(n),
            [node_of(c) for c in n._children],
            [tangents.get(id(c)) for c in n._children],
        )
    return tangents


class _coordinates:
    # the Jacobian from the columns: J[b, a] = coordinate b of the algebra
    # element H_a (algebra_kernels.rows, transposed), per grid and otype
    def __init__(self, grid, otype):
        self.grid = grid
        self.cartesian = otype.cartesian()
        self.kernels = g.group.algebra_kernels(grid, self.cartesian)
        self.otype = otype.jacobian_otype()
        c = container(g.lattice, grid, self.otype)
        self.op = primitive(
            "jacobian", self._plain, lambda *h: c.copy(), joint_vjp=self._vjp, order=1
        )

    def matrix(self, H):
        cols = []
        for h in H:
            if h is None:
                x = g.lattice(self.grid, self.cartesian)
                x[:] = 0
            else:
                x = g.lattice(self.grid, self.cartesian)
                x @= h
            cols.append(x)
        M = g.lattice(self.grid, self.otype)
        self.kernels.rows(M, cols)
        return M

    def _plain(self, *H):
        return g(g.transpose(self.matrix(H)))

    def flows(self, Jbar):
        # the flows into the columns, sum_b Jbar[b, a] T_b / n
        r = [g.lattice(self.grid, self.cartesian) for _ in range(self.kernels.ng)]
        self.kernels.combine(r, g(g.transpose(Jbar)))
        for x in r:
            x *= -1.0 / self.kernels.norm
        return r

    def _vjp(self, z, needed, *H):
        r = self.flows(z.gradient)
        return {a: r[a] for a in needed}


_coordinates_cache = {}


def _coordinates_of(x):
    c = x._container
    grid, otype = c.get_grid(), c.get_otype()
    for name in ["jacobian_directions", "jacobian_coordinates", "jacobian_otype"]:
        if not hasattr(otype, name):
            raise NotImplementedError(f"jacobian: the otype {otype.__name__} has no {name}")
    key = (grid, otype.__name__)
    if key not in _coordinates_cache:
        _coordinates_cache[key] = _coordinates(grid, otype)
    return _coordinates_cache[key], otype


def _columns(y, x, order, generators, otype, node_of=None):
    # the algebra elements whose coordinates are the columns of the chunk
    node_of = (lambda n: n) if node_of is None else node_of
    t = _tangents(order, x, otype.jacobian_directions(node_of(x), generators), node_of)
    dy = t[id(y)]
    return otype.jacobian_coordinates(node_of(y), dy)


class _chunked:
    # J as one primitive of the primal nodes (see chunk above)
    def __init__(self, y, x, chunk, coordinates, otype):
        self.y, self.x, self.otype = y, x, otype
        self.coordinates = coordinates
        self.order = _subgraph(y, x)
        primal, seen = [], set()
        for n in self.order:
            for c in [n] + list(n._children):
                if id(c) not in seen:
                    seen.add(id(c))
                    primal.append(c)
        self.primal = primal
        T = coordinates.kernels.field_generators
        self.chunks = [(a, T[a : a + chunk]) for a in range(0, len(T), chunk)]
        c = container(g.lattice, coordinates.grid, coordinates.otype)
        self.op = primitive(
            "jacobian(chunked)", self._plain, lambda *v: c.copy(), joint_vjp=self._vjp, order=1
        )

    def _leaves(self, values, needed=()):
        # internal leaves of the primal values (raw flows: no conversion), in
        # the nodes' own containers (a container conversion holds a value of
        # another type)
        leaves = {}
        for i, (n, v) in enumerate(zip(self.primal, values)):
            leaf = g.ad.reverse.node(v, with_gradient=i in needed, infinitesimal_to_cartesian=False)
            leaf._container = n._container.copy()
            leaves[id(n)] = leaf
        return leaves

    def _chunk_columns(self, leaves, generators):
        return _columns(self.y, self.x, self.order, generators, self.otype, lambda n: leaves[id(n)])

    def _plain(self, *values):
        coordinates = self.coordinates
        ng = coordinates.kernels.ng
        M = g.lattice(coordinates.grid, coordinates.otype)
        M[:] = 0
        for a, T in self.chunks:
            leaves = self._leaves(values)
            H = g.ad.reverse.linear.stack(self._chunk_columns(leaves, T))(with_gradients=False)
            M += coordinates.matrix([None] * a + list(H) + [None] * (ng - a - len(H)))
        return g(g.transpose(M))

    def _vjp(self, z, needed, *values):
        r = self.coordinates.flows(z.gradient)
        flows = {}
        for a, T in self.chunks:
            leaves = self._leaves(values, needed)
            L = None
            for i, h in enumerate(self._chunk_columns(leaves, T)):
                t = g.inner_product(constant(r[a + i]), h)
                L = t if L is None else L + t
            g.component.real(L).backward()
            for i in needed:
                gr = leaves[id(self.primal[i])].gradient
                if gr is not None:
                    flows[i] = gr if i not in flows else g(flows[i] + gr)
        return flows


def jacobian(y, x, chunk=None):
    # J(s) = d y(s) / d x(s) as a node (see above)
    coordinates, otype = _coordinates_of(x)
    T = coordinates.kernels.field_generators
    if chunk is not None and chunk < len(T):
        c = _chunked(y, x, chunk, coordinates, otype)
        return c.op(*c.primal)
    return coordinates.op(*_columns(y, x, _subgraph(y, x), T, otype))


# ---- per-site matrix functions of J (any square matrix otype), self-similar


def _inv_vjp(i, flow, M):
    # d M^-1 = -M^-1 dM M^-1: the flow into M is -adj(M^-1) flow adj(M^-1)
    Mi = inv(M)
    return -1.0 * (g.adj(Mi) * flow * g.adj(Mi))


inv = primitive(
    "matrix_inv",
    lambda M: g.matrix.inv(M),
    lambda c: c.copy(),
    vjp=_inv_vjp,
)


def _log_det_container(c):
    # (the type of g.matrix.det: a complex field)
    return container(g.lattice, c.get_grid(), g.complex(c.get_grid()).otype)


# d log det M = tr(M^-1 dM): the flow into M is flow adj(M^-1)
log_det = primitive(
    "matrix_log_det",
    lambda M: g(g.component.log(g.matrix.det(M))),
    _log_det_container,
    vjp=lambda i, flow, M: flow * g.adj(inv(M)),
)

# d det M = det M tr(M^-1 dM): the flow into M is flow conj(det M) adj(M^-1)
det = primitive(
    "matrix_det",
    lambda M: g.matrix.det(M),
    _log_det_container,
    vjp=lambda i, flow, M: flow * g.ad.reverse.transform.conj(det(M)) * g.adj(inv(M)),
)
