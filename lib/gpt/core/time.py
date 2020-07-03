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
import cgpt
import gpt

t0 = cgpt.time()


def time():
    return cgpt.time() - t0


class timer:
    def __init__(self, names):
        self.dt = {key: 0.0 for key in names}

    def start(self, which):
        self.dt[which] -= time()

    def stop(self, which):
        self.dt[which] += time()

    def print(self, prefix):
        for k, v in sorted(self.dt.items(), key=lambda x: x[1]):
            if "total" in self.dt:
                gpt.message(
                    "Timing %s: %15s = %e s (= %6.2f %%)"
                    % (prefix, k, v, v / self.dt["total"] * 100)
                )
            else:
                gpt.message("Timing %s: %15s = %e s" % (prefix, k, v))
