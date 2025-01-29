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


class symplectic_base:
    def __init__(self, name):
        self.__name__ = name
        self.scheme = []

    def add(self, op, step, direction):
        self.scheme.append((op, step, direction))

    def simplify(self):
        found = False
        prev_scheme = self.scheme
        while True:
            self.scheme = []
            for op, step, direction in prev_scheme:
                if abs(step) > 1e-15:
                    new_scheme = (op, step, 0)
                    if len(self.scheme) > 0 and self.scheme[-1][0] == op:
                        self.scheme[-1] = (op, step + self.scheme[-1][1], 0)
                        found = True
                    else:
                        self.scheme.append(new_scheme)
                else:
                    found = True

            if not found:
                break

            prev_scheme = self.scheme
            found = False

    def __str__(self):
        r = f"{self.__name__}"
        for op, step, direction in self.scheme:
            if isinstance(op, int):
                tag = f"I{op}"
            else:
                assert isinstance(op, tuple)
                tag = op[-1]
            r = r + f"\n  {tag}({step}, {direction})"
        return r

    def insert(self, *ip):
        scheme = []
        for op, step, direction in self.scheme:
            i = ip[op]
            if isinstance(i, symplectic_base):
                for op2, step2, direction2 in i.scheme:
                    scheme.append((op2, step * step2, 0))
            else:
                scheme.append((i, step, 0))
        self.scheme = scheme

    def unwrap(self):
        assert len(self.scheme) == 1
        op, step, direction = self.scheme[0]
        assert step == +1 and direction == 0
        assert isinstance(op, tuple)
        return op[1], op[2]

    def add_directions(self):
        n = len(self.scheme)
        if n % 2 == 1:
            pos = (n - 1) // 2
            if isinstance(self.scheme[pos], tuple) and not self.scheme[pos][0][-2]:
                i, step, direction = self.scheme[pos]
                mid = [(i, step / 2, +1), (i, step / 2, -1)]
                self.scheme = self.scheme[0:pos] + mid + self.scheme[pos + 1 :]

        n = len(self.scheme)
        for pos in range(n // 2):
            if isinstance(self.scheme[pos], tuple) and not self.scheme[pos][0][-2]:
                i, step, direction = self.scheme[pos]
                self.scheme[pos] = (i, step, +1)
                i, step, direction = self.scheme[n - pos - 1]
                self.scheme[n - pos - 1] = (i, step, -1)

    def __call__(self, tau):
        verbose = gpt.default.is_verbose(self.__name__)

        time = gpt.timer(f"Symplectic integrator {self.__name__}")
        time(self.__name__)

        n = len(self.scheme)
        for i in range(n):
            op, step, direction = self.scheme[i]
            if isinstance(op, int):
                raise Exception("Integrator not completely defined")
            else:
                assert isinstance(op, tuple)

                if verbose:
                    gpt.message(
                        f"{self.__name__} on step {i}/{n}: {op[-1]}({step * tau}, {direction})"
                    )

                op[1](step * tau, direction)

        if verbose:
            time()
            gpt.message(time)
