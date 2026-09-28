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
# Expression nodes: one node for a sum of products of its inputs,
#
#     z = sum_t c_t F_t1 F_t2 ... F_tn,    F = x_i or adj(x_i),
#
# built by the node arithmetic (*, +, -, adj, / and * by numbers) instead of
# one node per operation.  An operand that is an expression node whose value
# has not been computed is absorbed (its terms are copied; the operand itself
# stays a valid node, so a second use recomputes it, as for plain
# expressions); an operand with a computed value (e.g. a retained forward
# value) or of another kind is an input.
#
# The flow into input x_i sums, over all occurrences F_tk = x_i,
#
#     conj(c_t) adj(F_t1 ... F_t(k-1)) W adj(F_t(k+1) ... F_tn)
#
# (and the adjoint of it for F_tk = adj(x_i)), which is again a sum of
# products: evaluated with one g.eval for plain flows, and built with the
# same node arithmetic, i.e. as an expression node one level down, for nested
# (node) flows.  The derivative of an expression node is an expression node.
#
# Containers: operands are converted by the calling operation (see
# convert_container in __mul__) before they get here, so every input's flow
# has the input's container.
#
import gpt as g
import operator
from functools import reduce
from gpt.ad.reverse.util import value_of

# switch for A/B comparisons
enabled = True

# expansion limits (beyond them the operands stay inputs)
max_terms = 16
max_factors = 8


def _matrix_like(otype):
    # singlets and square matrices (also spin x color): any product order of
    # such factors is typed, so terms can be expanded and adjoints reversed
    # freely (a row vector times a matrix, e.g., has no multiplication table
    # entry)
    sh = otype.shape
    if sh == (1,):
        return True
    if len(sh) == 2:
        return sh[0] == sh[1]
    if len(sh) == 4:
        return sh[0] == sh[1] and sh[2] == sh[3]
    return False


_NUMBER, _FACTOR, _OTHER = 1, 2, 3


def _kind(x):
    # a number (coefficient), a factor (lattice or constant tensor of a
    # supported otype), or other; cached on the node
    k = x.__dict__.get("_ek")
    if k is None:
        if x._forward is None and g.util.is_num(x.value):
            k = _NUMBER
        elif (
            x._container.tag[0] == g.lattice
            or (x._forward is None and not x.with_gradient and isinstance(x.value, g.tensor))
        ) and _matrix_like(x._container.get_otype()):
            k = _FACTOR
        else:
            k = _OTHER
        x._ek = k
    return k


def _input(x):
    # (children, terms, max factors) of x as an input or a coefficient
    if x._ek == _NUMBER:
        return (), ((complex(x.value), ()),), 0
    return (x,), ((1.0, ((0, False),)),), 1


def _operand(x, op):
    # an absorbed expression (its value is not computed yet), a coefficient,
    # or an input.  As for plain expressions, a sum inside a product is
    # evaluated first (it is an input): products are never distributed over
    # sums
    t = x.__dict__.get("_terms")
    if t is not None and x.value is None and (op != "*" or len(t) == 1):
        return x._children, t, x._nmax
    return _input(x)


def _merge(ca, cb):
    # children of both operands (by identity) and the index map for b
    # (None: b's indices are unchanged)
    if not ca:
        return list(cb), None
    children = list(ca)
    remap = []
    for c in cb:
        for i, d in enumerate(children):
            if d is c:
                remap.append(i)
                break
        else:
            remap.append(len(children))
            children.append(c)
    return children, remap


def _terms_of(op, parts):
    if op == "*":
        a, b = parts
        return tuple((ca * cb, fa + fb) for ca, fa in a for cb, fb in b)
    if op == "adj":
        (a,) = parts
        return tuple(
            (complex(c).conjugate(), tuple((i, not adj) for i, adj in reversed(f))) for c, f in a
        )
    a, b = parts
    sign = 1.0 if op == "+" else -1.0
    return tuple(a) + tuple((sign * c, f) for c, f in b)


def _size(op, info):
    # (number of terms, max factors) of the result
    if op == "*":
        (_, ta, na), (_, tb, nb) = info
        return len(ta) * len(tb), na + nb
    if op == "adj":
        return len(info[0][1]), info[0][2]
    (_, ta, na), (_, tb, nb) = info
    return len(ta) + len(tb), max(na, nb)


def combine(op, operands, container):
    # an expression node for op(operands), or None (not an expression case)
    if not enabled or container.tag[0] != g.lattice:
        return None
    n_numbers = 0
    for x in operands:
        k = _kind(x)
        if k == _OTHER or (k == _NUMBER and op != "*"):
            return None
        n_numbers += k == _NUMBER
    if n_numbers == len(operands):
        return None
    info = [_operand(x, op) for x in operands]
    nt, nf = _size(op, info)
    if nt > max_terms or nf > max_factors:
        # keep the operands as inputs
        info = [_input(x) for x in operands]
        nt, nf = _size(op, info)
    children, remap = (), None
    parts = []
    for cx, tx, _ in info:
        children, remap = _merge(children, cx)
        if remap is not None:
            tx = tuple((c, tuple((remap[i], a) for i, a in f)) for c, f in tx)
        parts.append(tx)
    if len(children) == 0:
        return None
    return node(children, _terms_of(op, parts), nf, container)


def _is_node(v):
    return isinstance(v, g.ad.reverse.node_base)


def _nodify_values(values, extra=()):
    # nested case (some value or the flow is a node): plain values become
    # constant nodes, so that the arithmetic stays in the node world
    if not any(_is_node(v) for v in list(values) + list(extra)):
        return values
    return [v if _is_node(v) else g.ad.reverse.node_base(v, with_gradient=False) for v in values]


def _factor(v, adj):
    return g.adj(v) if adj else v


def _product(factors):
    return reduce(operator.mul, factors)


def _sum(parts):
    return reduce(operator.add, parts)


def _evaluate(values, terms):
    # sum_t c_t prod F (a plain g.expr, or a node for node values)
    parts = []
    for c, f in terms:
        p = _product([_factor(values[i], a) for i, a in f])
        parts.append(p if c == 1.0 else c * p)
    return _sum(parts)


def _flow(values, terms, i, w):
    # the flow into input i for the flow w into the node (see the header)
    parts = []
    for c, f in terms:
        for k, (j, a) in enumerate(f):
            if j != i:
                continue
            left = [_factor(values[n], not b) for n, b in reversed(f[:k])]
            right = [_factor(values[n], not b) for n, b in reversed(f[k + 1 :])]
            if not a:
                # conj(c) adj(left) w adj(right)
                p = _product(left + [w] + right)
                cc = complex(c).conjugate()
            else:
                # the adjoint: c right' adj(w) left' with right' = adj of the
                # right factors in original order, left' likewise
                p = _product(
                    [_factor(values[n], b) for n, b in f[k + 1 :]]
                    + [g.adj(w)]
                    + [_factor(values[n], b) for n, b in f[:k]]
                )
                cc = complex(c)
            parts.append(p if cc == 1.0 else cc * p)
    return _sum(parts)




def _flows_plain(values, terms, occurrences, need, w):
    # all input flows of a plain node at once.  In a term F_0 ... F_{n-1}
    # with several gradient-carrying occurrences, the flow at occurrence k,
    # X_k = adj(F_0 ... F_{k-1}) w adj(F_{k+1} ... F_{n-1}) = L_k Q_k, is
    # built from running products from both ends,
    #   Q_{n-1} = w,  Q_{k-1} = Q_k adj(F_k)      (materialized)
    #   L_0 = 1,      L_{k+1} = adj(F_k) L_k      (materialized, L_1 lazy)
    # i.e. about 3n products per term instead of n(n-1); each input's
    # contributions are summed in one g.eval
    parts = {}
    for (c, f), occ in zip(terms, occurrences):
        ks = [o for o in occ if o[1] in need]
        if not ks:
            continue
        n = len(f)
        if n == 1:
            X = {0: w}
        else:
            F = [_factor(values[j], a) for j, a in f]
            if len(ks) == 1 or n == 2:
                X = {}
                for k, j, a in ks:
                    X[k] = _product(
                        [g.adj(F[m]) for m in reversed(range(k))]
                        + [w]
                        + [g.adj(F[m]) for m in reversed(range(k + 1, n))]
                    )
            else:
                kmin = ks[0][0]
                kmax = ks[-1][0]
                Q = {n - 1: w}
                for k in range(n - 1, kmin, -1):
                    Q[k - 1] = g(Q[k] * g.adj(F[k]))
                L = {0: None, 1: g.adj(F[0])}
                for k in range(1, kmax):
                    L[k + 1] = g(g.adj(F[k]) * L[k])
                X = {k: Q[k] if L[k] is None else L[k] * Q[k] for k, j, a in ks}
        for k, j, a in ks:
            if not a:
                cc = complex(c).conjugate()
                p = X[k]
            else:
                cc = complex(c)
                p = g.adj(X[k])
            parts.setdefault(j, []).append(p if cc == 1.0 else cc * p)
    return {i: g(_sum(p)) for i, p in parts.items()}


def node(children, terms, nmax, container):
    children = tuple(children)
    # occurrences (position, input, adjoint) per term, for the backward
    occurrences = tuple(tuple((k, j, a) for k, (j, a) in enumerate(f)) for c, f in terms)
    # nested (node-valued) or plain: fixed by the depth of the inputs,
    # decided at the first backward
    nested = [None]

    def _forward():
        # (node.forward evaluates a plain expression; a lone adj stays lazy)
        return _evaluate(_nodify_values([value_of(c) for c in children]), terms)

    def _backward(z):
        w = z.gradient
        values = [value_of(c) for c in children]
        if nested[0] is None or _is_node(w):
            nested[0] = _is_node(w) or any(_is_node(v) for v in values)
        need = {i for i, c in enumerate(children) if c.with_gradient}
        if nested[0]:
            # nested: the flows are expression nodes one level down
            values = _nodify_values(values, (w,))
            flows = {i: _flow(values, terms, i, w) for i in need}
        else:
            flows = _flows_plain(values, terms, occurrences, need, w)
        for i, v in flows.items():
            g.ad.reverse.util.accum(children[i], v, 1)

    n = g.ad.reverse.node_base(_forward, _backward, children, _container=container, _tag="expr")
    n._terms = terms
    n._nmax = nmax
    return n
