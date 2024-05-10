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
import gpt
import os
import struct
import numpy


class checkpointer_none:
    def __init__(self):
        self.grid = None

    def save(self, obj):
        pass

    def load(self, obj):
        return False


class checkpointer:
    def __init__(self, root):
        self.root = root
        self.grid = None
        directory = "%s/%2.2d" % (root, gpt.rank() // 32)
        os.makedirs(directory, exist_ok=True)
        self.filename = "%s/%10.10d" % (directory, gpt.rank())
        try:
            self.f = gpt.FILE(self.filename, "r+b")
        except FileNotFoundError:
            self.f = gpt.FILE(self.filename, "w+b")
        self.f.seek(0, 1)
        self.verbose = gpt.default.is_verbose("checkpointer")

    def save(self, obj):
        if isinstance(obj, list):
            for o in obj:
                self.save(o)
        elif isinstance(obj, gpt.lattice):
            self.save(obj.mview())
        elif isinstance(obj, float):
            self.save(memoryview(struct.pack("d", obj)))
        elif isinstance(obj, complex):
            self.save(memoryview(struct.pack("dd", obj.real, obj.imag)))
        elif isinstance(obj, memoryview):
            self.f.seek(0, 1)
            sz = len(obj)
            szGB = sz / 1024.0**3
            self.f.write(sz.to_bytes(8, "little"))
            t0 = gpt.time()
            self.f.write(gpt.crc32(obj).to_bytes(4, "little"))
            t1 = gpt.time()
            self.f.write(obj)
            self.f.flush()
            t2 = gpt.time()
            if self.verbose:
                if self.grid is None:
                    gpt.message(
                        "Checkpoint %g GB on head node at %g GB/s for crc32 and %g GB/s for write in %g s total"
                        % (szGB, szGB / (t1 - t0), szGB / (t2 - t1), t2 - t0)
                    )
                else:
                    szGB = self.grid.globalsum(szGB)
                    gpt.message(
                        "Checkpoint %g GB at %g GB/s for crc32 and %g GB/s for write in %g s total"
                        % (szGB, szGB / (t1 - t0), szGB / (t2 - t1), t2 - t0)
                    )
        else:
            assert 0

    def load(self, obj):
        if isinstance(obj, list):
            if len(obj) != 1:
                allok = True
                pos = self.f.tell()
                for i, o in enumerate(obj):
                    r = [o]
                    allok = allok and self.load(r)
                    obj[i] = r[0]
                if not allok:
                    self.f.seek(pos, 0)  # reset position to overwrite corrupted data chunk
                return allok
            else:
                if isinstance(obj[0], gpt.lattice):
                    res = self.load(obj[0].mview())
                elif isinstance(obj[0], float):
                    v = memoryview(bytearray(8))
                    res = self.load(v)
                    obj[0] = struct.unpack("d", v)[0]
                elif isinstance(obj[0], complex):
                    v = memoryview(bytearray(16))
                    res = self.load(v)
                    obj[0] = complex(*struct.unpack("dd", v)[0, 1])
                elif isinstance(obj[0], memoryview):
                    return self.read_view(obj[0])
                else:
                    assert 0
                return res
        elif isinstance(obj, memoryview):
            return self.read_view(obj)
        elif isinstance(obj, gpt.lattice):
            return self.load(obj.mview())
        else:
            assert 0

    def read_view(self, obj):
        pos = self.f.tell()
        self.f.seek(0, 2)
        flags = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float64)
        t0 = gpt.time()
        if self.f.tell() != pos:
            self.f.seek(pos, 0)
            # try to read
            sz = int.from_bytes(self.f.read(8), "little")
            szGB = sz / 1024.0**3
            flags[2] = szGB
            crc32_expected = int.from_bytes(self.f.read(4), "little")
            if len(obj) == sz:
                data = self.f.read(sz)
                if len(data) == sz:
                    obj[:] = data
                    crc32 = gpt.crc32(obj)
                    if crc32 == crc32_expected:
                        flags[0] = 1.0  # flag success on this node

        # compare global
        assert self.grid is not None
        self.grid.globalsum(flags)
        t1 = gpt.time()

        # report status
        if self.verbose and flags[2] != 0.0:
            if flags[0] != flags[1]:
                gpt.message(
                    "Checkpoint %g GB failed on %g out of %g nodes"
                    % (flags[2], flags[1] - flags[0], flags[1])
                )
            else:
                gpt.message(
                    "Checkpoint %g GB at %g GB/s for crc32 and read combined in %g s total"
                    % (flags[2], flags[2] / (t1 - t0), t1 - t0)
                )

        # all nodes OK?
        if flags[0] == flags[1]:
            return True

        # reset position to overwrite corruption
        self.f.seek(pos, 0)

        return False
