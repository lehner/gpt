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
# Diagnostics of a function and its training:
#
#   ref = g.ml.snapshot(net)                       # before training
#   ...
#   g.ml.displacement(net, ref)                    # how far each parameter moved
#   g.ml.activity(net, [x])                        # how much each call changes its input
#   g.ml.gradient_noise(make_cost, fields, n=8)    # signal-to-noise of stochastic gradients
#
import gpt as g
from gpt.ad.reverse.util import get_container


def _norm2(x):
    if isinstance(x, list):
        return sum(_norm2(y) for y in x)
    if g.util.is_num(x):
        return abs(x) ** 2
    return g.norm2(x)


def _difference(a, b):
    if isinstance(a, list):
        return [_difference(x, y) for x, y in zip(a, b)]
    if g.util.is_num(a):
        return a - b
    return g(a - b)


def _copy(x):
    return x if g.util.is_num(x) else g.copy(x)


def snapshot(f):
    """Copies of the parameters of f by name (a reference for displacement)."""
    return dict(zip(f.parameter_names(), [_copy(x) for x in f.parameters()]))


def displacement(f, reference):
    """{name: (|p - p_ref|, |p - p_ref| / |p_ref|)} for the parameters of f
    against a snapshot (the relative change is None for p_ref = 0).  Moves
    of the order of the optimizer's step size only, or relative changes far
    below one, mean the parameter is still where it was initialized."""
    result = {}
    for name, p in zip(f.parameter_names(), f.parameters()):
        d = _norm2(_difference(p, reference[name])) ** 0.5
        r = _norm2(reference[name]) ** 0.5
        result[name] = (d, d / r if r > 0 else None)
    return result


def activity(f, inputs):
    """How much each call of a composite (or the function itself) changes
    its input, on plain inputs: {call name: |y - x| / |x|} for the first
    output y and the first input x of the same type (the residual branch
    relative to the skip connection; lists summed over their elements), None
    if no input has the type of y (e.g. replicate).  The entry "" is the
    function as a whole (its first output against its first input of the
    same type).  A residual block with a ratio far below the others (or
    below the size of the effect it should model) is still close to the
    identity."""
    from gpt.ml.graph import composite

    def ratio(xs, y):
        t = get_container(y)
        for x in xs:
            if get_container(x) == t:
                return (_norm2(_difference(y, x)) / max(_norm2(x), 1e-300)) ** 0.5
        return None

    result = {}
    if isinstance(f, composite):

        def after(c, values):
            result[c.name] = ratio([values[s.id] for s in c.inputs], values[c.outputs[0].id])

        output = f._replay(inputs, f.parameters(), after)[0]
    else:
        output = f(inputs)[0]
    result[""] = ratio(inputs, output)
    return result


def gradient_noise(cost, fields, n, names=None):
    """The signal-to-noise ratio of a stochastic gradient: cost() returns a
    new draw of the cost functional (e.g. with new random directions), all
    at the current fields.  For each field: {name: (|mean|, std, snr)} with
    the mean and the standard deviation (per draw) of its gradient over n
    draws and snr = |mean|^2 / std^2.  Below about one the sign of a single
    draw's gradient is mostly noise (Adam then performs a random walk of the
    size of its step): average more draws per step, lower the step size, or
    use a lower-variance estimator.  The mean of n draws has std / sqrt(n)."""
    fields = g.util.to_list(fields)
    names = [str(i) for i in range(len(fields))] if names is None else names
    draws = [cost().gradient(fields, fields) for _ in range(n)]
    result = {}
    for i, name in enumerate(names):
        gs = [d[i] for d in draws]
        if g.util.is_num(gs[0]):
            mean = sum(gs) / n
        else:
            mean = g(sum(gs[1:], gs[0]) * (1.0 / n))
        var = sum(_norm2(_difference(x, mean)) for x in gs) / max(n - 1, 1)
        m2 = _norm2(mean)
        result[name] = (m2**0.5, var**0.5, m2 / var if var > 0 else float("inf"))
    return result
