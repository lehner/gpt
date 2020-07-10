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
        self.f = {}
        self.b = {}
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
                self.f[which] = 0.0
                self.b[which] = 0.0
            self.current = which
            self.f[which] += flop if flop is not None else 0.0
            self.b[which] += byte if byte is not None else 0.0
            self.dt[which] -= time()
        else:
            self.dt["total"] += time()
            self.active = False

    def print(self):
        dtp, fp, bp = self.create_print_arrays()

        if dtp["total"] != 0.0:
            for k, v in sorted(dtp.items(), key=lambda x: x[1]):
                if fp["total"] != 0.0 or bp["total"] != 0.0:
                    gpt.message(
                        "%s: profiling: %15s = %e s (= %6.2f %%) %e F/s %e B/s"
                        % (
                            self.prefix,
                            k,
                            v,
                            v / dtp["total"] * 100,
                            fp[k] / v,
                            bp[k] / v,
                        )
                    )
                else:
                    gpt.message(
                        "%s: timing: %15s = %e s (= %6.2f %%)"
                        % (self.prefix, k, v, v / dtp["total"] * 100)
                    )

    def create_print_arrays(self):
        """
        Return copies with enhancements we don't want to have in the raw collected data
        """
        dtp, fp, bp = self.dt.copy(), self.f.copy(), self.b.copy()

        if "total" in dtp:
            total = dtp["total"]
            profiled = sum(dtp.values()) - total
            dtp["unprofiled"] = total - profiled
        else:
            dtp["total"] = sum(dtp.values())
            dtp["unprofiled"] = 0.0  # by construction

        fp["total"] = sum(fp.values())
        bp["total"] = sum(bp.values())
        fp["unprofiled"] = 0.0
        bp["unprofiled"] = 0.0

        return dtp, fp, bp
