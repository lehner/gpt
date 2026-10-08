#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023-2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
# Differentiable matrix exponential as a self-similar tower of fused kernels.
#
# The node exp(X) is the k=0 member of the family
#
#     D_k(X; H_1, ..., H_k) = d^k exp_X (H_1, ..., H_k),
#
# the k-th directional derivative (symmetric, multilinear in the H_i).  Every
# D_k is holomorphic in its arguments, and the reverse-mode flows of D_k are
# again members of the family (W = the flow into D_k):
#
#     flow into X   :  D_{k+1}(X^dag; H_1^dag, ..., H_k^dag, W)
#     flow into H_i :  D_k(X^dag; H_1^dag, ..., W, ..., H_k^dag)   (W at slot i)
#
# i.e., the adjoint of a derivative of exp is a derivative of exp at X^dag.
# (For k=0 this is the familiar int_0^1 e^{s X^dag} W e^{(1-s) X^dag} ds.)
# So, as for stencils, the gradient of exp is exp: a recorded pass builds D_k
# nodes, and at the plain level each D_k is ONE compiled local
# kernel instead of a graph of O(100) elementwise node operations.
#
# All members are JETS: exp(X + sum_b eps_b H_b) with eps_b^2 = 0, restricted
# to a downward-closed family of components (subsets S of the
# infinitesimals; the S component is D_|S|(X; H_S)), the outputs sums of
# components, ONE fused kernel per jet.  D_k is the jet of all subsets with
# the top component as its output.  The rule above for a jet: for the flow
# W_j into output O_j = sum_{S in L_j} comp_S, with a new infinitesimal d_j
# along W_j,
#
#     flow into X   = sum_j sum_{S in L_j} comp_{S + d_j}
#     flow into H_b = sum_j sum_{S in L_j, b in S} comp_{S - b + d_j}
#
# at X^dag (directions H^dag), again one jet: the flows into X and all H_b in
# one kernel.  Forward tangents (g.ad.reverse.jacobian) along X are new
# infinitesimals as well: dO_j along T_a = sum_{S in L_j} comp_{S + e_a}, all
# outputs and directions in one kernel.
#
# Plain evaluation: gpt.core.foundation.lattice.matrix.exp.jet (exp of a
# sparse multi-dual number, one compiled kernel per jet).
#
import gpt as g
from gpt.ad.reverse.primitive import primitive
from gpt.ad.reverse.util import container


def _plain_base(x):
    # x = L or x = adj(L) for a plain lattice L: (L, adjoint?); else (None, None).
    # Flows at X^dag are the lazy expression adj(X) (also when built at node
    # level, where adj stays unevaluated), and adj(adj(X)) is X
    if isinstance(x, g.lattice):
        return x, False
    if isinstance(x, g.expr) and x.unary == g.expr_unary.NONE and len(x.val) == 1:
        coef, term = x.val[0]
        if coef == 1.0 and len(term) == 1 and isinstance(term[0][1], g.lattice):
            unary = term[0][0]
            if unary == g.factor_unary.NONE:
                return term[0][1], False
            if unary == g.factor_unary.ADJ:
                return term[0][1], True
    return None, None


class _tower:
    # the plain X shared by all jets of one exp tower (the node exp(X) and
    # the flows and tangents built from it, to any order): every plain jet is
    # evaluated at X or X^dag, which have the same scaling s, so s and the
    # materialized X^dag are computed once.  X is identified by identity; the
    # user-created root resets the tower in every pass (see derivative), so
    # a leaf modified in place between evaluations is never served stale
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = None
        self.s = None
        self.xdag = None

    def base(self, x):
        # (the plain lattice of x = X or X^dag, its scaling); a value that is
        # neither is evaluated, with its own scaling
        base, dag = _plain_base(x)
        if base is None:
            return g(x), None
        if base is not self.x:
            self.x = base
            self.s = g.lattice.foundation.matrix.exp.scaling(base)
            self.xdag = None
        if dag:
            if self.xdag is None:
                self.xdag = g(g.adj(base))
            base = self.xdag
        return base, self.s


def _closure(components):
    # the downward closure (all subsets) of the components
    family = set()
    for S in components:
        family.update(g.lattice.foundation.matrix.exp._submasks(S))
    return sorted(family)


def _plain(x, *h, family, outputs, listed, tower, root):
    # (fused kernels live in the lattice foundation)
    if root:
        tower.reset()
    base, s = tower.base(x)
    out = g.lattice.foundation.matrix.exp.jet(base, list(h), family, outputs, s)
    return out if listed else out[0]


def _fwd(x, *h, **static):
    # a node's plain value; the residual records that the root has reset the
    # tower in this pass
    return _plain(x, *h, **static), True


def _vjp(z, needed, x, *h, family, outputs, listed, tower, root, residual):
    # one jet at X^dag for all needed flows (see above).  A root whose value
    # was not computed in this pass (with_value=False) resets the tower here,
    # before its first use
    if root and residual is None:
        tower.reset()
    flows = z.gradient if listed else [z.gradient]
    nb = len(h)
    dirs = [g.adj(y) for y in h]
    to_x, to_h = [], [[] for _ in range(nb)]
    for j, L in enumerate(outputs):
        if flows[j] is None:
            continue
        d = 1 << len(dirs)
        dirs.append(flows[j])
        for S in L:
            to_x.append(S | d)
            for b in range(nb):
                if S & (1 << b):
                    to_h[b].append((S & ~(1 << b)) | d)
    out, owners = [], []
    if 0 in needed and to_x:
        out.append(to_x)
        owners.append(0)
    for b in range(nb):
        if (b + 1) in needed and to_h[b]:
            out.append(to_h[b])
            owners.append(b + 1)
    if not out:
        return {}
    r = jet(g.adj(x), dirs, _closure([S for L in out for S in L]), out, tower)
    return {o: r[i] for i, o in enumerate(owners)}


def _jvp(z, children, tangents, family, outputs, listed, tower, root):
    # the forward tangents along X: a new infinitesimal per direction,
    # dO_j = sum_{S in L_j} comp_{S + e_a}, all in one jet
    if any(t is not None for t in tangents[1:]):
        raise NotImplementedError("jacobian: tangents of the directions of exp derivatives")
    t = tangents[0]
    k, nb = len(t), len(children) - 1
    e = [1 << (nb + a) for a in range(k)]
    out = [[S | ea for S in L] for ea in e for L in outputs]
    r = jet(children[0], list(children[1:]) + list(t), _closure([S for L in out for S in L]), out, tower)
    m = len(outputs)
    if not listed:
        return [r[a] for a in range(k)]
    return [g.ad.reverse.linear.stack([r[a * m + j] for j in range(m)]) for a in range(k)]


def _container(x, *h, family, outputs, listed, tower, root):
    return container(list, x, len(outputs)) if listed else x


_J = primitive("exp_jet", _plain, _container, joint_vjp=_vjp, fwd=_fwd, jvp=_jvp)


def jet(x, h, family, outputs, tower=None, listed=True):
    # the outputs (lists of subset masks of the infinitesimals along h) of
    # exp(x + sum_b eps_b h_b) restricted to the downward-closed family; x
    # may be adj(X) of the tower's X.  Directions outside the family are
    # dropped.  Returns a list (plain) or a list node; listed=False: the
    # single output itself.  Without a tower, the jet is the user's root of
    # a new tower
    used = 0
    for S in family:
        used |= S
    bits = [b for b in range(len(h)) if used & (1 << b)]

    def remap(S):
        return sum(1 << i for i, b in enumerate(bits) if S & (1 << b))

    root = tower is None
    return _J(
        x,
        *[h[b] for b in bits],
        family=sorted(remap(S) for S in family),
        outputs=[[remap(S) for S in L] for L in outputs],
        listed=listed,
        tower=_tower() if root else tower,
        root=root,
    )


def derivative(x, h, tower=None):
    # D_k(x; h_1..h_k): the jet of all subsets, the top component; plain
    # values run the fused kernel, node values build a node whose backward is
    # again a jet, in the same tower.  The user-created root resets the tower
    # in every pass (when it computes a plain value, or else in its
    # backward), so a leaf modified in place between passes is never served
    # stale
    k = len(h)
    return jet(x, h, list(range(2**k)), [[2**k - 1]], tower, listed=False)


def function(x):
    if x._container.tag[0] != g.lattice:
        # non-lattice (tensor / scalar) values: the generic scaling-and-
        # squaring Taylor series of the forward AD, as a graph of node ops
        return g.ad.forward.foundation.matrix.exp.function(x)
    return derivative(x, [])
