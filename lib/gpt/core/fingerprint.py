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
fingerprint_paused = False
fingerprint_status_time = 0.0
fingerprint_global_timer = None

def pause():
    global fingerprint_paused
    fingerprint_paused = True

def unpause():
    global fingerprint_paused
    fingerprint_paused = False

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

def flush():
    global fingerprint_file
    if fingerprint_file is not None:
        fingerprint_file.flush()
        


class log:
    def __init__(self):
        global fingerprint_global_timer
        
        if fingerprint_paused:
            return

        if fingerprint_global_timer is None:
            fingerprint_global_timer = g.timer("fingerprint.log")
        fingerprint_global_timer("backtrace.getframe")
        frame = sys._getframe(1)
        stack = ""
        fingerprint_global_timer("backtrace.format")
        while frame is not None:
            stack = f"{stack}{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}\n"
            frame = frame.f_back
        self.stack = stack
        self.messages = []
        fingerprint_global_timer()

    def __call__(self, first=None, second=None):
        global fingerprint_file, fingerprint_index, fingerprint_status_time, fingerprint_global_timer

        if fingerprint_paused:
            return
        
        if second is not None:
            if isinstance(second, np.ndarray):
                self.messages.append((first, np.copy(second)))
            elif isinstance(second, list):
                for i, x in enumerate(second):
                    self(f"{first}.{i}", x)
            elif isinstance(second, g.lattice):
                # create fingerprint
                tag = f"{second.otype.__name__}.{second.grid}"
                if tag not in fingerprints:
                    fingerprint_global_timer("create fingerprint field")
                    fingerprints[tag] = g.random(tag).cnormal(g.lattice(second))
                    fingerprint_global_timer()

                fingerprint_global_timer("compute fingerprint field")
                fp = [g.rank_inner_product(fingerprints[tag], second)]
                fingerprint_global_timer()
                
                self(first, np.array(fp, dtype=np.complex128))
            else:
                self(first, np.array(g.util.to_list(second), dtype=np.complex128))

        else:

            if fingerprint_file is None:
                return

            fingerprint_global_timer("write fingerprint")
            fingerprint_file.write(f"Log {fingerprint_index}:\n{self.stack}")
            for a, b in self.messages:
                fingerprint_file.write(f"Entry {a}:\n")
                #fingerprint_file.write(f"Type: {type(b)}\n")
                #fingerprint_file.write(f"Value: {b}\n")
                fingerprint_global_timer("write fingerprint.numpy")
                np.savetxt(fingerprint_file, b)
                fingerprint_global_timer("write fingerprint")
            fingerprint_file.write("\n")
            fingerprint_index += 1
            fingerprint_global_timer()

            tm = g.time()
            if tm > fingerprint_status_time + 30:
                fingerprint_status_time = tm
                fingerprint_file.flush()
                g.message(fingerprint_global_timer)
