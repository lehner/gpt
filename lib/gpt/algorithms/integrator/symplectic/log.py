#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-25  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Mattia Bruno
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
import gpt


class log:
    def __init__(self):
        self.grad = {}
        self.time = gpt.timer()
        self.verbose = gpt.default.is_verbose("symplectic_log")
        self.last_log = None

    def reset(self):
        self.time = gpt.timer()
        for key in self.grad:
            self.grad[key] = []

    def gradient(self, gs, name):
        if name not in self.grad:
            self.grad[name] = []

        self.time("norm")
        gn = 0.0
        v = 0
        for g in gpt.core.util.to_list(gs):
            gn += gpt.norm2(g)
            v += g.grid.gsites
        self.time()
        self.grad[name].append(gn / v)
        if self.verbose:
            if self.last_log is None:
                self.last_log = gpt.time()
            if gpt.time() - self.last_log > 10:
                self.last_log = gpt.time()
                gpt.message("Force status")
                gpt.message(self.time)

    def __call__(self, grad, name):
        def inner():
            self.time(name)
            gs = grad()
            self.time()
            self.gradient(gs, name)
            return gs

        return inner

    def get(self, key):
        return self.grad[key]
