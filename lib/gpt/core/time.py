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
    def __init__(self):
        self.dt = {}

    def start(self, which):
        if which not in self.dt:
            self.dt[which] = 0.0
        self.dt[which] -= time()

    def stop(self, which):
        self.dt[which] += time()

    def print(self, prefix):
        to_print = (
            self.dt.copy()
        )  # don't want to have additions below in raw collected data

        if "total" in to_print:
            total = to_print["total"]
            profiled = sum(to_print.values()) - total
            to_print["unprofiled"] = total - profiled
        else:
            to_print["total"] = sum(to_print.values())
            to_print["unprofiled"] = 0.0  # by construction

        for k, v in sorted(to_print.items(), key=lambda x: x[1]):
            gpt.message(
                "Timing %s: %15s = %e s (= %6.2f %%)"
                % (prefix, k, v, v / to_print["total"] * 100)
            )
