#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt, sys


class base:
    def __init__(self, name=None):
        self.name = self.__class__.__name__ if name is None else name
        self.verbose = gpt.default.is_verbose(self.name)
        self.verbose_debug = gpt.default.is_verbose(self.name + "_debug")
        self.verbose_performance = gpt.default.is_verbose(self.name + "_performance")
        self.timer = gpt.timer(self.name, self.verbose_performance)

    def timed_start(self):
        return gpt.timer(self.name) if self.verbose_performance else self.timer

    def timed_end(self, t):
        if self.verbose_performance:
            t()
            self.timer += t
            gpt.message(
                f"\nPerformance of {self.name}:\n\nThis call:\n{t}\n\nAll calls:\n{self.timer}\n"
            )

    def timed_function(self, function):
        def timed_function(*a, **b):
            t = self.timed_start()
            ret = function(*a, t, **b)
            self.timed_end(t)
            return ret

        return timed_function

    def timed_method(function):
        def timed_function(self, *a, **b):
            t = self.timed_start()
            ret = function(self, *a, t, **b)
            self.timed_end(t)
            return ret

        return timed_function

    def log(self, *a):
        if self.verbose:
            gpt.message(f"{self.name}:", *a)

    def debug(self, *a):
        if self.verbose_debug:
            gpt.message(f"{self.name}:", *a)


class base_iterative(base):
    def __init__(self, name=None):
        super().__init__(name)
        self.verbose_convergence = gpt.default.is_verbose(self.name + "_convergence")

    def log_convergence(self, iteration, value, target=None):
        if (type(iteration) == int and iteration == 0) or (
            type(iteration) == tuple and all([x == 0 for x in iteration])
        ):
            self.history = []
        self.history.append(value)
        if self.verbose_convergence:
            if target is None:
                # expect to log a value that can be converted to a string
                gpt.message(f"{self.name}: iteration {iteration}: {value}")
            else:
                # expect residual
                gpt.message(
                    f"{self.name}: iteration {iteration}: {value:e} / {target:e}"
                )
