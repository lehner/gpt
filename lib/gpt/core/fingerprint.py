#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import sys, socket, os
import numpy as np
from inspect import getframeinfo

fingerprints = {}
fingerprint_file = None
fingerprint_index = 0


def start(tag):
    global fingerprint_file, fingerprint_index
    if g.default.get_int("--fingerprint", 0) < 1:
        return

    # make sure we all agree on the tag
    tag = g.broadcast(0, tag)
    
    if g.rank() == 0:
        if not os.path.exists(tag):
            os.makedirs(tag)
    g.barrier()

    fingerprint_index = 0
    fingerprint_file = open(f"{tag}/fingerprint.{g.rank()}", "wt")
    fingerprint_file.write(f"Host: {socket.gethostname()}\n\n")
    fingerprint_file.write(f"Environment: {dict(os.environ)}\n\n")


class log:
    def __init__(self):
        frame = sys._getframe(1)
        stack = ""
        while frame is not None:
            caller = getframeinfo(frame)
            stack = f"{stack}{caller.filename}:{caller.lineno}\n"
            frame = frame.f_back
        self.stack = stack
        self.messages = []

    def __call__(self, first=None, second=None):
        global fingerprint_file, fingerprint_index

        if second is not None:
            if isinstance(second, np.ndarray):
                self.messages.append((first, np.copy(second)))
            elif isinstance(second, list):
                for i, x in enumerate(second):
                    self(f"{first}.{i}", x)
            elif isinstance(second, g.lattice):
                # create fingerprint
                fp = []
                for y in x:
                    tag = f"{y.otype.__name__}.{y.grid}"
                    if tag not in fingerprints:
                        fingerprints[tag] = g.random(tag).cnormal(g.lattice(y))
                    fp.append(g.rank_inner_product(fingerprints[tag], y))
                self(first, np.array(fp, dtype=np.complex128))
            else:
                self(first, np.array(g.util.to_list(second), dtype=np.complex128))

        else:

            if fingerprint_file is None:
                return

            fingerprint_file.write(f"Log {fingerprint_index}:\n{self.stack}")
            for a, b in self.messages:
                fingerprint_file.write(f"Entry {a}:\n")
                #fingerprint_file.write(f"Type: {type(b)}\n")
                #fingerprint_file.write(f"Value: {b}\n")
                np.savetxt(fingerprint_file, b)
            fingerprint_file.write("\n")
            fingerprint_file.flush()
            fingerprint_index += 1


