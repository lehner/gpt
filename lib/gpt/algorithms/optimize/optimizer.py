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
from gpt.algorithms import base_iterative


class optimizer(base_iterative):
    """The calling conventions of the optimizers:

      opt.on(x, dx=None)(f)                 iterations with the functional f on
                                            the fields x, updating the subset
                                            dx (default: all of x)
      opt(f)(x, dx)                         the same with f given first

    A run created by opt.on keeps the optimizer's state (e.g. Adam's moments)
    across its calls, also with a different f in each call (e.g. a cost drawn
    anew for every step, with maxiter=1 at construction).  opt(f) keeps one
    state for all its calls.  Subclasses implement new_state() and
    iterate(f, x, dx_indices, state, t)."""

    def on(self, x, dx=None):
        return _run(self, x, dx, self.new_state())

    def __call__(self, f):
        state = self.new_state()

        def opt(x, dx):
            return _run(self, x, dx, state)(f)

        return opt

    def new_state(self):
        return None

    def iterate(self, f, x, dx_indices, state, t):
        raise NotImplementedError()


class _run:
    # an optimizer bound to fields x and the updated subset dx (by position
    # in x, which stays valid when numbers in x are replaced by updates)
    def __init__(self, opt, x, dx, state):
        self.opt = opt
        self.x = g.util.to_list(x)
        dx = self.x if dx is None else g.util.to_list(dx)
        self.dx_indices = [g.util.index_by_identity(self.x, y) for y in dx]
        self.state = state

    def __call__(self, f):
        t = self.opt.timed_start()
        result = self.opt.iterate(f, self.x, self.dx_indices, self.state, t)
        self.opt.timed_end(t)
        return result
