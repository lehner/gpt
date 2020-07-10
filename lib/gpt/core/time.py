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
    def __init__(self, prefix):
        self.dt = {}
        self.flop = {}
        self.byte = {}
        self.prefix = prefix
        self.active = False
        self.current = None

    def __call__(self, which=None, *, flop=None, byte=None):
        """
        first started timer also starts total timer
        with argument which given starts a new timer and ends a previous one if running
        without argument which ends current + total timer
        """
        if self.active is False and which is not None:
            if "total" not in self.dt:
                self.dt["total"] = 0.0
            self.active = True
            self.dt["total"] -= time()

        if self.current is not None:
            self.dt[self.current] += time()
            self.current = None

        if which is not None:
            if which not in self.dt:
                self.dt[which] = 0.0
                self.flop[which] = 0.0
                self.byte[which] = 0.0
            self.current = which
            self.flop[which] += flop if flop is not None else 0.0
            self.byte[which] += byte if byte is not None else 0.0
            self.dt[which] -= time()
        else:
            self.dt["total"] += time()
            self.active = False

    def print(self):
        dt_print, flop_print, byte_print = (
            self.dt.copy(),
            self.flop.copy(),
            self.byte.copy(),
        )  # don't want to have additions below in raw collected data

        if "total" in dt_print:
            total = dt_print["total"]
            profiled = sum(dt_print.values()) - total
            dt_print["unprofiled"] = total - profiled
        else:
            dt_print["total"] = sum(dt_print.values())
            dt_print["unprofiled"] = 0.0  # by construction

        flop_print["total"] = sum(flop_print.values())
        byte_print["total"] = sum(byte_print.values())
        flop_print["unprofiled"] = 0.0
        byte_print["unprofiled"] = 0.0

        if dt_print["total"] != 0.0:
            for k, v in sorted(dt_print.items(), key=lambda x: x[1]):
                if flop_print["total"] != 0.0 or byte_print["total"] != 0.0:
                    gpt.message(
                        "%s: profiling: %15s = %e s (= %6.2f %%) %e F/s %e B/s"
                        % (
                            self.prefix,
                            k,
                            v,
                            v / dt_print["total"] * 100,
                            flop_print[k] / v,
                            byte_print[k] / v,
                        )
                    )
                else:
                    gpt.message(
                        "%s: timing: %15s = %e s (= %6.2f %%)"
                        % (self.prefix, k, v, v / dt_print["total"] * 100)
                    )
