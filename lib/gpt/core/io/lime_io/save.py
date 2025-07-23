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
import sys, cgpt, os, struct
import numpy as np
import xml.etree.ElementTree as ET
from gpt.core.io.util import distribute_cartesian_file
import gpt.core.io.lime_io.scidac as scidac


class lime_writer:
    magic = 1164413355

    def __init__(self, fn, comm):
        self.fn = fn
        self.comm = comm

        if self.comm.processor == 0:
            # first create and truncate file
            f = g.FILE(fn, "wb")
            del f

        self.comm.barrier()

        # then all ranks reopen in same mode
        #self.f = g.FILE(fn, "r+b")

        #self.comm.barrier()

        self.append_offset = 0

        self.index = {}

    def close(self):
        self.f = None

    def reopen(self):
        self.f = g.FILE(self.fn, "r+b")

    def create_tag(self, tag, size):
        size = int(size)
        if self.comm.processor == 0:
            self.reopen()
            self.f.seek(self.append_offset, 0)
            self.f.write(struct.pack(">L", self.magic))
            self.f.write(struct.pack(">H", 1))
            self.f.write(struct.pack(">H", 0b1111111111111111))
            self.f.write(struct.pack(">Q", size))
            bin_tag = (tag + "\0").encode("utf-8")
            assert len(bin_tag) <= 128
            bin_tag = bin_tag + ("\0" * (128 - len(bin_tag))).encode("utf-8")
            self.f.write(bin_tag)

            self.f.seek(0, 1)
            offset = int(self.f.tell())
            self.close()

        else:
            offset = 0
            size = 0

        offset = self.comm.globalsum(offset)
        size = self.comm.globalsum(size)

        if size % 8 != 0:
            dsize = 8 - size % 8
        else:
            dsize = 0

        self.append_offset = offset + size + dsize

        self.index[tag] = (offset, size)

    def write(self, tag, element_offset, data):
        assert tag in self.index
        tag_offset, size = self.index[tag]
        assert element_offset + len(data) <= size

        self.reopen()
        self.f.seek(tag_offset + element_offset, 0)
        self.f.write(data)
        self.close()

    def write_text(self, tag, text):
        assert isinstance(text, str)
        text = text.encode("utf-8")
        self.create_tag(tag, len(text))
        if self.comm.processor == 0:
            self.write(tag, 0, text)

    def _xml_build(self, root, xml):
        root = ET.Element(root)
        for e in xml:
            if isinstance(xml[e], dict):
                root.append(self._xml_build(e, xml[e]))
            else:
                ET.SubElement(root, e).text = str(xml[e])
        return root

    def write_xml(self, tag, root, xml):
        root = self._xml_build(root, xml)
        self.write_text(
            tag, ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
        )


def save(file, objects, params):
    objects = g.util.to_list(objects)
    assert all([isinstance(x, g.lattice) for x in objects])
    size = objects[0].global_bytes()
    assert all([x.global_bytes() == size for x in objects])
    grid = objects[0].grid

    w = lime_writer(file, grid)
    for et in params["tag_order"]:
        w.write_text(et, params["tags"][et])

    w.create_tag(params["binary_data_tag"], size * len(objects))

    verbose = g.default.is_verbose("io")

    # performance
    dt_distr, dt_crc, dt_write, dt_misc = 0.0, 0.0, 0.0, 0.0
    szGB = 0.0
    grid.barrier()
    t0 = g.time()
    dt_write -= g.time()

    grid = objects[0].grid
    cb = objects[0].checkerboard()
    pos, nwriter = distribute_cartesian_file(grid.fdimensions, grid, cb)
    ssize = objects[0].global_bytes() // grid.gsites
    size = ssize * len(objects)

    scidac_checksum_a = np.uint32(0)
    scidac_checksum_b = np.uint32(0)

    # distributes data accordingly
    data_munged = memoryview(bytearray(len(objects) * len(pos) * ssize))
    dt_distr -= g.time()
    for mu in range(len(objects)):
        data_munged[mu * len(pos) * ssize : (mu + 1) * len(pos) * ssize] = g.mview(objects[mu][pos])
    grid.barrier()
    dt_distr += g.time()

    if len(pos) > 0:

        dt_misc -= g.time()
        data = memoryview(bytearray(len(data_munged)))
        cgpt.munge_inner_outer(
            data,
            data_munged,
            len(pos),
            len(objects),
        )
            
        if sys.byteorder != "big":
            cgpt.munge_byte_order(data, data, 8)

        dt_misc += g.time()

        dt_crc -= g.time()
        scidac_checksum_a, scidac_checksum_b = scidac.checksums(data, grid, pos)
        dt_crc += g.time()
            
        sz = size * len(pos)
        w.write(params["binary_data_tag"], grid.processor * sz, data)
        szGB += len(data) / 1024.0**3.0

    grid.barrier()

    dt_write += g.time()

    crc_comp_a, crc_comp_b = scidac.checksums_reduce(
        scidac_checksum_a, scidac_checksum_b, nwriter, grid
    )
    crc_comp_a = f"{crc_comp_a:8x}"
    crc_comp_b = f"{crc_comp_b:8x}"

    w.write_xml(
        "scidac-checksum",
        "scidacChecksum",
        {"version": "1.0", "suma": crc_comp_a, "sumb": crc_comp_b},
    )

    w.write_xml(
        "gpt-format",
        "gptFormat",
        {"grid": grid.describe(), "n": len(objects), "otype": objects[0].describe()},
    )

    grid.barrier()
    t1 = g.time()

    szGB = grid.globalsum(szGB)
    if verbose and dt_crc != 0.0:
        g.message(
            "Write %g GB at %g GB/s (%g GB/s for distribution, %g GB/s for writing + checksum, %g GB/s for checksum, %d writers)"
            % (
                szGB,
                szGB / (t1 - t0),
                szGB / dt_distr,
                szGB / dt_write,
                szGB / dt_crc,
                nwriter,
            )
        )
