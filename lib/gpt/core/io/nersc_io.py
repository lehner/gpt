#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2021  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import cgpt, gpt, numpy, os, sys
from gpt.core.io.util import distribute_cartesian_file


class nersc_io:
    def __init__(self, path):
        self.path = path
        self.fdimensions = []
        self.bytes_header = -1
        self.metadata = {}
        self.verbose = gpt.default.is_verbose("io")
        gpt.barrier()

    def read_header(self):

        # make sure this is a file
        if not os.path.isfile(self.path):
            return False

        with open(self.path, "rb") as f:
            line = self.getline(f)
            if line != "BEGIN_HEADER":
                return False

            # need to be mute before this line since this is used to autodetect the file format
            if self.verbose:
                gpt.message(f"NERSC file format; reading {self.path}")
                gpt.message(f"   {line}")

            line = self.getline(f)
            while line != "END_HEADER":
                if self.verbose:
                    gpt.message(f"\t{line}")
                [field, val] = line.split("=", 1)
                self.metadata[field.strip()] = val.strip()
                line = self.getline(f)

            if self.verbose:
                gpt.message(f"   {line}")

            self.bytes_header = f.tell()
            f.seek(0, 2)
            self.bytes_data = f.tell() - self.bytes_header

        self.fdimensions = [int(self.metadata[f"DIMENSION_{i+1}"]) for i in range(4)]
        self.floating_point = self.metadata["FLOATING_POINT"]
        self.data_type = self.metadata["DATATYPE"]

        if self.floating_point == "IEEE64BIG":
            self.munge = self.munge_ieee64big
            self.precision = gpt.double
        elif self.floating_point == "IEEE64LITTLE" or self.floating_point == "IEEE64":
            self.munge = self.munge_ieee64little
            self.precision = gpt.double
        else:
            gpt.message("Warning: unknown floating point format {self.floating_point}")
            return False

        gsites = int(numpy.prod(self.fdimensions))
        assert self.bytes_data % gsites == 0
        self.bytes_per_site = self.bytes_data // gsites

        assert self.bytes_per_site % self.precision.nbytes == 0
        self.floats_per_site = self.bytes_per_site // self.precision.nbytes

        if self.data_type == "4D_SU3_GAUGE_3x3":
            assert 3 * 3 * 4 * 2 == self.floats_per_site
            self.otype = gpt.ot_matrix_su_n_fundamental_group(3)
            self.nfields = 4
            self.reconstruct = self.reconstruct_none
        elif self.data_type == "4D_SU3_GAUGE":
            assert 3 * 2 * 4 * 2 == self.floats_per_site
            self.otype = gpt.ot_matrix_su_n_fundamental_group(3)
            self.reconstruct = self.reconstruct_third_row
            self.nfields = 4
        else:
            gpt.message("Warning: unknown data type {self.data_type}")
            return False

        assert self.floats_per_site % self.nfields == 0
        self.floats_per_field_site = self.floats_per_site // self.nfields
        return True

    def getline(self, f):
        return f.readline().decode("utf-8").strip()

    def munge_ieee64big(self, data):
        if sys.byteorder == "little":
            cgpt.munge_byte_order(data, data, 8)
        return data

    def munge_ieee64little(self, data):
        if sys.byteorder == "big":
            cgpt.munge_byte_order(data, data, 8)
        return data

    def reconstruct_none(self, data):
        return data

    def reconstruct_third_row(self, data):
        sz = len(data)
        assert sz % 2 == 0
        sz = sz // 2 * 3
        assert sz % 8 == 0
        data_munged = cgpt.mview(cgpt.ndarray([sz // 8], numpy.float64))
        cgpt.munge_reconstruct_third_row(data_munged, data, self.precision.nbytes)
        return data_munged

    def read_lattice(self):
        # define grid from header
        g = gpt.grid(self.fdimensions, self.precision)
        # create lattice
        l = [gpt.lattice(g, self.otype) for i in range(self.nfields)]

        # performance
        dt_distr, dt_cs, dt_read, dt_misc = 0.0, 0.0, 0.0, 0.0
        szGB = 0.0
        g.barrier()
        t0 = gpt.time()

        dt_read -= gpt.time()

        pos, nreader = distribute_cartesian_file(
            self.fdimensions, g, l[0].checkerboard()
        )

        if len(pos) > 0:
            sz = self.bytes_per_site * len(pos)
            f = gpt.FILE(self.path, "rb")
            f.seek(self.bytes_header + g.processor * sz, 0)
            data = memoryview(f.read(sz))
            f.close()

            dt_misc -= gpt.time()
            data = self.munge(data)
            dt_misc += gpt.time()

            dt_cs -= gpt.time()
            cs_comp = cgpt.util_nersc_checksum(data, 0)
            dt_cs += gpt.time()

            dt_misc -= gpt.time()
            data = self.reconstruct(data)

            assert len(data) % 8 == 0
            data_munged = cgpt.mview(cgpt.ndarray([len(data) // 8], numpy.float64))
            cgpt.munge_inner_outer(data_munged, data, self.nfields, len(pos))
            data = data_munged
            dt_misc += gpt.time()

            szGB += len(data) / 1024.0 ** 3.0
        else:
            data = memoryview(bytearray())
            cs_comp = 0

        cs_comp = g.globalsum(cs_comp) & 0xFFFFFFFF
        cs_exp = int(self.metadata["CHECKSUM"].upper(), 16)
        if cs_comp != cs_exp:
            gpt.message(f"cs_comp={cs_comp:X} cs_exp={cs_exp:X}")
            assert False

        dt_read += gpt.time()

        # distributes data accordingly
        g.barrier()
        dt_distr -= gpt.time()
        cache = {}
        lblock = len(data) // self.nfields
        for i in range(self.nfields):
            l[i][pos, cache] = data[lblock * i : lblock * (i + 1)]
        g.barrier()
        dt_distr += gpt.time()

        g.barrier()
        t1 = gpt.time()

        szGB = g.globalsum(szGB)
        if self.verbose and dt_cs != 0.0:
            gpt.message(
                "Read %g GB at %g GB/s (%g GB/s for distribution, %g GB/s for munged read, %g GB/s for checksum, %g GB/s for munging, %d readers)"
                % (
                    szGB,
                    szGB / (t1 - t0),
                    szGB / dt_distr,
                    szGB / dt_read,
                    szGB / dt_cs,
                    szGB / dt_misc,
                    nreader,
                )
            )

        # also check plaquette and link trace
        P_comp = gpt.qcd.gauge.plaquette(l)
        P_exp = float(self.metadata["PLAQUETTE"])
        P_digits = len(self.metadata["PLAQUETTE"].split(".")[1])
        P_eps = abs(P_comp / P_exp - 1.0)
        P_eps_threshold = 10.0 ** (-P_digits + 1)
        P_eps_threshold = max([1e-9, P_eps_threshold])
        assert P_eps < P_eps_threshold

        L_comp = (
            sum(
                [gpt.sum(gpt.trace(x)) / x.grid.gsites / x.otype.shape[0] for x in l]
            ).real
            / self.nfields
        )
        L_exp = float(self.metadata["LINK_TRACE"])
        L_digits = len(self.metadata["LINK_TRACE"].split(".")[1].lower().split("e")[0])
        L_eps_threshold = 10.0 ** (-L_digits + 1)
        L_eps_threshold = max([1e-9, L_eps_threshold])
        L_eps = abs(L_comp / L_exp - 1.0)
        assert L_eps < L_eps_threshold

        return l


def load(filename, p={}):

    lat = nersc_io(filename)

    # check if this is right file format from header
    if not lat.read_header():
        raise NotImplementedError()

    return lat.read_lattice()
