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
import gpt as g
from gpt.ad.reverse.primitive import primitive


def _solve(op, parameters, rhs, solver):
    # the plain solution of op(*parameters, w) = rhs; op reads the parameter
    # values at call time, the solver sees a plain linear map
    def mat(dst, src):
        dst @= op(*parameters, src)

    return solver(g.matrix_operator(mat=mat, accept_guess=(True, False)))(rhs)


def _fwd(*args, op, adj_op, solver):
    w = _solve(op, args[:-1], args[-1], solver)
    return w, w


def _joint_vjp(z, needed, *args, op, adj_op, solver, residual):
    # w = A(p)^-1 b.  For the flow c into w: lambda = A^-dag c is the flow
    # into b, and dw = -A^-1 (dA) w gives the flow into p as that of
    # op(*p, w) at fixed w for the flow -lambda (one plain reverse pass over a
    # graph of op with fresh leaves for the parameters that need a flow)
    parameters, rhs = list(args[:-1]), args[-1]
    n = len(parameters)
    # (no residual: the forward did not run in this pass, with_value=False)
    w = _solve(op, parameters, rhs, solver) if residual is None else residual
    lam = _solve(op if adj_op is None else adj_op, parameters, z.gradient, solver)
    result = {}
    if n in needed:
        result[n] = lam
    flowed = [i for i in needed if i < n]
    if flowed:
        leaves = [
            g.ad.reverse.node(p, with_gradient=i in flowed, infinitesimal_to_cartesian=False)
            for i, p in enumerate(parameters)
        ]
        y = op(*leaves, g.ad.reverse.node(w, with_gradient=False))
        y.backward(initial_gradient=g(-1.0 * lam))
        for i in flowed:
            result[i] = leaves[i].gradient
    return result


_linear_solve = primitive(
    "linear_solve",
    lambda *args, op, adj_op, solver: _fwd(*args, op=op, adj_op=adj_op, solver=solver)[0],
    lambda *c, op, adj_op, solver: c[-1].copy(),
    joint_vjp=_joint_vjp,
    fwd=_fwd,
    order=1,
)


def linear_solve(op, rhs, solver, parameters=[], adj_op=None):
    """The solution w of op(*parameters, w) = rhs, with the GPT inverter
    solver, as a node if rhs or a parameter is a node (plain otherwise).

    op(*parameters, x) is linear in x and works on plain fields (the solver
    iterations) and on nodes (the parameter flows: the operator is
    differentiated by AD).  adj_op(*parameters, x) is its adjoint, used for
    the solve of the backward; None: op is hermitian (e.g. for cg).  The
    gradient is that of the exact inverse: calling linear_solve declares the
    solve to be converged (the gradient error is of the order of the solver
    tolerance).  First derivatives only."""
    if isinstance(op, g.matrix_operator) or isinstance(adj_op, g.matrix_operator):
        raise NotImplementedError("linear_solve: op and adj_op must be functions, not matrix_operators")
    return _linear_solve(*parameters, rhs, op=op, adj_op=adj_op, solver=solver)
