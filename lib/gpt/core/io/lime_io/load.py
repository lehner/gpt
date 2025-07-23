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


class lime_reader:
    magic = 1164413355

    @classmethod
    def is_file(cls, fn):
        if not os.path.isfile(fn):
            return False
        f = g.FILE(fn, "rb")
        magic = f.read(4)
        if len(magic) == 0:
            return False
        return struct.unpack(">L", magic)[0] == cls.magic

    def __init__(self, fn):
        self.fn = fn
        self.f = g.FILE(fn, "rb")

        self.index = {}

        while True:
            magic = self.f.read(4)
            if len(magic) == 0:
                break
            assert struct.unpack(">L", magic)[0] == self.magic
            version = struct.unpack(">H", self.f.read(2))[0]
            assert version == 1
            self.f.read(2)  # reserved
            size = struct.unpack(">Q", self.f.read(8))[0]
            tag = self.f.read(128).decode("utf-8").strip("\0")
            offset = self.f.tell()
            self.f.seek(size, 1)
            if size % 8 != 0:
                self.f.seek(8 - size % 8, 1)

            self.index[tag] = (offset, size)

    def seek(self, tag, element_offset):
        assert tag in self.index
        tag_offset, size = self.index[tag]
        assert element_offset < size
        self.f.seek(tag_offset + element_offset, 0)
        return size

    def read(self, size):
        return self.f.read(size)

    def get_text(self, tag):
        size = self.seek(tag, 0)
        return self.read(size).decode("utf-8").strip("\0")

    def get_xml(self, tag):
        return ET.fromstring(self.get_text(tag))

    def get_xml_dict(self, tag):
        ret = {}
        for child in self.get_xml(tag):
            ct = child.tag
            if ct[0] == "{":
                ct = ct[ct.index("}") + 1 :]
            ret[ct] = child.text
        return ret

    def allocate_fields(self, params):

        # Allow for liberal assignment of data to target buffer
        if "target" in params:
            return params["target"]

        # ILDG lattice QCD gauge fields; keep support for domain-specific versions to a minimum
        if "ildg-format" in self.index:
            d = self.get_xml_dict("ildg-format")
            L = [int(d["lx"]), int(d["ly"]), int(d["lz"]), int(d["lt"])]
            assert d["field"] == "su3gauge"

            # define grid from header
            if d["precision"] == "64":
                grid = g.grid(L, g.double)
            elif d["precision"] == "32":
                grid = g.grid(L, g.single)
            else:
                raise TypeError(f"Unknown precision in lime file: {d['precision']}")

            # create lattice
            return [g.mcolor(grid) for mu in range(4)]

        elif "gpt-format" in self.index:
            d = self.get_xml_dict("gpt-format")
            grid = g.grid_from_description(d["grid"])
            n = int(d["n"])
            return [g.lattice(grid, d["otype"]) for _ in range(n)]

        keys = list(self.index.keys())
        raise TypeError(f"Unknown format of lime file: {keys}")

    def scidac_checksum(self):
        if "scidac-checksum" in self.index:
            return self.get_xml_dict("scidac-checksum")

        return None


def load(file, params):

    # check file format
    if not lime_reader.is_file(file):
        raise NotImplementedError()

    # open lime_reader
    r = lime_reader(file)

    # first get dimensions
    U = r.allocate_fields(params)
    checksum = r.scidac_checksum()
    if checksum is None:
        g.message(f"Warning: LIME file {file} does not provide a checksum!")

    verbose = g.default.is_verbose("io")

    # performance
    dt_distr, dt_crc, dt_read, dt_misc = 0.0, 0.0, 0.0, 0.0
    szGB = 0.0
    g.barrier()
    t0 = g.time()
    dt_read -= g.time()

    grid = U[0].grid
    cb = U[0].checkerboard()
    pos, nreader = distribute_cartesian_file(grid.fdimensions, grid, cb)
    ssize = U[0].global_bytes() // grid.gsites
    size = ssize * len(U)

    scidac_checksum_a = np.uint32(0)
    scidac_checksum_b = np.uint32(0)

    # find binary data tag
    binary_data_tag = None
    for bdt in r.index:
        if bdt[-12:] == "-binary-data":
            assert binary_data_tag is None
            binary_data_tag = bdt

    if binary_data_tag is None:
        raise TypeError("Did not find a *-binary-data tag to load")

    if len(pos) > 0:
        sz = size * len(pos)
        assert r.seek(binary_data_tag, grid.processor * sz) == grid.gsites * size

        data = memoryview(r.read(sz))

        dt_crc -= g.time()
        scidac_checksum_a, scidac_checksum_b = scidac.checksums(data, grid, pos)
        dt_crc += g.time()

        dt_misc -= g.time()
        if sys.byteorder != "big":
            cgpt.munge_byte_order(data, data, 8)

        data_munged = memoryview(bytearray(len(data)))
        cgpt.munge_inner_outer(
            data_munged,
            data,
            len(U),
            len(pos),
        )
        dt_misc += g.time()

        szGB += len(data) / 1024.0**3.0
    else:
        data_munged = memoryview(bytearray())

    g.barrier()
    dt_read += g.time()

    crc_comp_a, crc_comp_b = scidac.checksums_reduce(
        scidac_checksum_a, scidac_checksum_b, nreader, grid
    )
    crc_comp_a = f"{crc_comp_a:8x}"
    crc_comp_b = f"{crc_comp_b:8x}"

    if checksum is not None:
        assert checksum["suma"] == crc_comp_a
        assert checksum["sumb"] == crc_comp_b

    # distributes data accordingly
    dt_distr -= g.time()
    for mu in range(len(U)):
        U[mu][pos] = data_munged[mu * len(pos) * ssize : (mu + 1) * len(pos) * ssize]
    grid.barrier()
    dt_distr += g.time()

    grid.barrier()
    t1 = g.time()

    szGB = grid.globalsum(szGB)
    if verbose and dt_crc != 0.0:
        g.message(
            "Read %g GB at %g GB/s (%g GB/s for distribution, %g GB/s for reading + checksum, %g GB/s for checksum, %d readers)"
            % (
                szGB,
                szGB / (t1 - t0),
                szGB / dt_distr,
                szGB / dt_read,
                szGB / dt_crc,
                nreader,
            )
        )
    return g.util.from_list(U)
