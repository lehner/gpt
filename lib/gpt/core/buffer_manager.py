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
import numpy as np


class accelerator_buffer_manager:
    def __init__(self):
        self.avail = {}
        self.stats = {"new": 0, "reuse": 0}

    def request(self, shape, dtype):
        tag = (int(np.prod(shape)), dtype)
        if tag in self.avail and len(self.avail[tag]) > 0:
            self.stats["reuse"] += 1
            return self.avail[tag].pop().reshape(shape)
        self.stats["new"] += 1
        return g.accelerator_buffer(shape=shape, dtype=dtype)

    def release(self, buf):
        tag = (int(np.prod(buf.shape)), buf.dtype)
        if tag not in self.avail:
            self.avail[tag] = []
        self.avail[tag].append(buf)

    def size(self):
        return sum(x.calculate_size(x.shape, x.dtype) for y in self.avail for x in self.avail[y])

    def __str__(self):
        return f"""
 Size : {self.size() / 1e9:.2e} GB
 New  : {self.stats["new"]}
 Reuse: {self.stats["reuse"]}
        """
