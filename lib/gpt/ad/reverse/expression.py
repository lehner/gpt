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


def _is_number_leaf(x):
    return x._forward is None and g.util.is_num(x.value)


def _is_constant_tensor_leaf(x):
    return x._forward is None and not x.with_gradient and isinstance(x.value, g.tensor)


def _is_lattice(x):
    return x._container.tag[0] == g.lattice


def _absorbable(x):
    return getattr(x, "_terms", None) is not None and x.value is None


def _operand(x, op):
    # (children, terms) of an operand: absorbed expression, coefficient, or
    # input.  As for plain expressions, a sum inside a product is evaluated
    # first (it is an input), products are never distributed over sums
    if _absorbable(x) and (op != "*" or len(x._terms) == 1):
        return list(x._children), list(x._terms)
    if _is_number_leaf(x):
        return [], [(complex(x.value), ())]
    return [x], [(1.0, ((0, False),))]


def _merge(ca, cb):
    # children list of both operands (by identity) and the index map for b
    children = list(ca)
    index = {id(c): i for i, c in enumerate(children)}
    remap = []
    for c in cb:
        if id(c) not in index:
            index[id(c)] = len(children)
            children.append(c)
        remap.append(index[id(c)])
    return children, remap


def _remap(terms, remap):
    return [(c, tuple((remap[i], a) for i, a in f)) for c, f in terms]


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


def _supported(x, allow_number):
    if _is_number_leaf(x):
        return allow_number
    if not (_is_lattice(x) or _is_constant_tensor_leaf(x)):
        return False
    return _matrix_like(x._container.get_otype())


def _combine(op, operands, container):
    ch, terms = [], []
    parts = []
    for x in operands:
        cx, tx = _operand(x, op)
        ch, remap = _merge(ch, cx)
        parts.append(_remap(tx, remap))
    if op == "*":
        a, b = parts
        terms = [(ca * cb, fa + fb) for ca, fa in a for cb, fb in b]
    elif op in ("+", "-"):
        a, b = parts
        s = 1.0 if op == "+" else -1.0
        terms = a + [(s * c, f) for c, f in b]
    elif op == "adj":
        (a,) = parts
        terms = [(complex(c).conjugate(), tuple((i, not adj) for i, adj in reversed(f))) for c, f in a]
    else:
        raise Exception(f"unknown expression operation {op}")
    return ch, terms


def _too_large(terms):
    return len(terms) > max_terms or any(len(f) > max_factors for c, f in terms)


def combine(op, operands, container):
    # an expression node for op(operands), or None (not an expression case)
    if not enabled or container.tag[0] != g.lattice:
        return None
    allow_number = op == "*"
    if not all(_supported(x, allow_number) for x in operands):
        return None
    if op == "*" and all(_is_number_leaf(x) for x in operands):
        return None
    children, terms = _combine(op, operands, container)
    if _too_large(terms):
        # keep the operands as inputs
        saved = [getattr(x, "_terms", None) for x in operands]
        children, terms = [], None
        parts = []
        for x in operands:
            if _is_number_leaf(x):
                parts.append([(complex(x.value), ())])
                continue
            children, remap = _merge(children, [x])
            parts.append([(1.0, ((remap[0], False),))])
        if op == "*":
            terms = [(ca * cb, fa + fb) for ca, fa in parts[0] for cb, fb in parts[1]]
        elif op in ("+", "-"):
            s = 1.0 if op == "+" else -1.0
            terms = parts[0] + [(s * c, f) for c, f in parts[1]]
        else:
            terms = [(complex(c).conjugate(), tuple((i, not a) for i, a in reversed(f))) for c, f in parts[0]]
    if len(children) == 0:
        return None
    return node(children, terms, container)


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




def _flows_plain(values, terms, need, w):
    # all input flows of a plain node at once.  In a term F_0 ... F_{n-1}
    # with several gradient-carrying occurrences, the flow at occurrence k,
    # X_k = adj(F_0 ... F_{k-1}) w adj(F_{k+1} ... F_{n-1}) = L_k Q_k, is
    # built from running products from both ends,
    #   Q_{n-1} = w,  Q_{k-1} = Q_k adj(F_k)      (materialized)
    #   L_0 = 1,      L_{k+1} = adj(F_k) L_k      (materialized, L_1 lazy)
    # i.e. about 3n products per term instead of n(n-1); each input's
    # contributions are summed in one g.eval
    parts = {i: [] for i in need}
    for c, f in terms:
        ks = [k for k, (j, a) in enumerate(f) if j in parts]
        if not ks:
            continue
        n = len(f)
        F = [_factor(values[j], a) for j, a in f]
        if len(ks) == 1 or n <= 2:
            X = {}
            for k in ks:
                X[k] = _product(
                    [g.adj(F[m]) for m in reversed(range(k))]
                    + [w]
                    + [g.adj(F[m]) for m in reversed(range(k + 1, n))]
                )
        else:
            Q = {n - 1: w}
            for k in range(n - 1, min(ks), -1):
                Q[k - 1] = g(Q[k] * g.adj(F[k]))
            L = {0: None, 1: g.adj(F[0])}
            for k in range(1, max(ks)):
                L[k + 1] = g(g.adj(F[k]) * L[k])
            X = {k: Q[k] if L[k] is None else L[k] * Q[k] for k in ks}
        for k in ks:
            j, a = f[k]
            if not a:
                cc = complex(c).conjugate()
                p = X[k]
            else:
                cc = complex(c)
                p = g.adj(X[k])
            parts[j].append(p if cc == 1.0 else cc * p)
    return {i: g(_sum(p)) for i, p in parts.items() if p}


def node(children, terms, container):
    children = tuple(children)

    def _forward():
        # (node.forward evaluates a plain expression; a lone adj stays lazy)
        return _evaluate(_nodify_values([value_of(c) for c in children]), terms)

    def _backward(z):
        w = z.gradient
        values = _nodify_values([value_of(c) for c in children], (w,))
        need = [i for i, c in enumerate(children) if c.with_gradient]
        if _is_node(w) or any(_is_node(v) for v in values):
            # nested: the flows are expression nodes one level down
            flows = {i: _flow(values, terms, i, w) for i in need}
        else:
            flows = _flows_plain(values, terms, need, w)
        for i, v in flows.items():
            g.ad.reverse.util.accum(children[i], v, 1)

    n = g.ad.reverse.node_base(_forward, _backward, children, _container=container, _tag="expr")
    n._terms = terms
    return n
