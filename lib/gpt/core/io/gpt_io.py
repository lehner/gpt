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
import cgpt, gpt, os, io, numpy, sys, fnmatch, glob
from gpt.params import params_convention

# get local dir an filename
def get_local_name(root, cv):
    if cv.rank < 0:
        return None, None
    ntotal = cv.ranks
    rank = cv.rank
    dirs = 32
    nperdir = ntotal // dirs
    if nperdir < 1:
        nperdir = 1
    dirrank = rank // nperdir
    directory = "%s/%2.2d" % (root, dirrank)
    filename = "%s/%10.10d.field" % (directory, rank)
    return directory, filename


# gpt io class
class gpt_io:
    @params_convention(mpi=None, grids=None, paths=None)
    def __init__(self, root, write, params):
        self.root = root
        self.params = params
        if self.params["grids"] is None:
            self.params["grids"] = {}
        self.verbose = gpt.default.is_verbose("io")

        if gpt.rank() == 0:
            os.makedirs(self.root, exist_ok=True)
            if write:
                self.glb = gpt.FILE(root + "/global", "wb")
                for f in glob.glob("%s/??/*.field" % self.root):
                    os.unlink(f)
            else:
                self.glb = gpt.FILE(root + "/global", "rb")
        else:
            self.glb = None

        self.loc = {}
        self.pos = {}
        self.loc_desc = ""
        self.cache = {}

        # If we write, keep an index buffer
        self.index_file = io.StringIO("") if write else None

        # now sync since only root has created directory
        gpt.barrier()

    def flush(self):
        # if we read, no need to flush
        if self.index_file is None:
            return

        # get memoryview of current index
        mvidx = memoryview(self.index_file.getvalue().encode("utf-8"))

        # write index to fs
        index_crc = gpt.crc32(mvidx)
        if gpt.rank() == 0:
            f = open(self.root + "/index", "wb")
            f.write(mvidx)
            f.close()
            f = open(self.root + "/index.crc32", "wt")
            f.write("%X\n" % index_crc)
            f.close()

    def close_views(self):
        for f in self.loc:
            if self.loc[f] is not None:
                self.loc[f].close()
        self.loc = {}
        self.pos = {}
        self.loc_desc = ""

    def close_global(self):
        if self.glb is not None:
            self.glb.close()
            self.glb = None

    def close(self):
        if self.index_file is not None:
            self.flush()
            self.index_file = None
            # no more writing after close
        self.close_global()
        self.close_views()

    def open_view(self, xk, iview, write, mpi, fdimensions, g_cb, l_cb):
        cv = gpt.cartesian_view(
            iview if iview is not None else -1, mpi, fdimensions, g_cb, l_cb
        )
        dn, fn = get_local_name(self.root, cv)
        loc_desc = cv.describe() + "/" + ("Write" if write else "Read")

        tag = "%d-%s" % (xk, str(iview))
        tag_pos = "%s-%s-%s-%s" % (tag, str(fdimensions), str(g_cb), str(l_cb))

        if loc_desc != self.loc_desc:
            self.close_views()
            self.loc_desc = loc_desc
            if self.verbose:
                gpt.message("Switching view to %s" % self.loc_desc)

        if tag not in self.loc:
            if write and dn is not None:
                os.makedirs(dn, exist_ok=True)
            self.loc[tag] = (
                gpt.FILE(fn, "a+b" if write else "rb") if fn is not None else None
            )

        if tag_pos not in self.pos:
            self.pos[tag_pos] = gpt.coordinates(cv)

        return self.loc[tag], self.pos[tag_pos]

    def views_for_node(self, cv, grid):
        return cv.views_for_node(grid)

    def write_lattice(self, ctx, l):
        g = l.grid
        tag = (ctx + "\0").encode("utf-8")
        ntag = len(tag)
        nd = len(g.fdimensions)

        # create cartesian view for writing
        if self.params["mpi"] is not None:
            mpi = self.params["mpi"]
        else:
            mpi = g.mpi
        cv0 = gpt.cartesian_view(-1, mpi, g.fdimensions, g.cb, l.checkerboard())

        # file positions
        pos = numpy.array([0] * cv0.ranks, dtype=numpy.uint64)

        # describe
        res = g.describe() + " " + cv0.describe() + " " + l.describe()

        # find tasks for my node
        views_for_node = self.views_for_node(cv0, g)

        # performance
        dt_distr, dt_crc, dt_write = 0.0, 0.0, 0.0
        # g.barrier()
        t0 = gpt.time()
        szGB = 0.0

        # need to write all views
        for xk, iview in enumerate(views_for_node):

            f, p = self.open_view(
                xk, iview, True, mpi, g.fdimensions, g.cb, l.checkerboard()
            )

            cache_key = res + f"_{g.obj}_{xk}_{iview}_{id(p)}_{l.otype.__name__}_write"
            if cache_key not in self.cache:
                self.cache[cache_key] = {}

            # all nodes are needed to communicate
            dt_distr -= gpt.time()
            mv = gpt.mview(l[p, self.cache[cache_key]])
            dt_distr += gpt.time()

            # write data
            if f is not None:
                # description and data
                dt_crc -= gpt.time()
                crc = gpt.crc32(mv)
                dt_crc += gpt.time()
                dt_write -= gpt.time()
                pos[iview] = f.tell()
                f.write(ntag.to_bytes(4, byteorder="little"))
                f.write(tag)
                f.write(crc.to_bytes(4, byteorder="little"))
                f.write(nd.to_bytes(4, byteorder="little"))
                for i in range(nd):
                    f.write(g.gdimensions[i].to_bytes(4, byteorder="little"))
                for i in range(nd):
                    f.write(g.mpi[i].to_bytes(4, byteorder="little"))
                f.write(len(mv).to_bytes(8, byteorder="little"))
                f.write(mv)
                f.flush()
                dt_write += gpt.time()
                szGB += len(mv) / 1024.0 ** 3.0

        t1 = gpt.time()

        szGB = g.globalsum(szGB)
        if self.verbose and dt_crc != 0.0:
            gpt.message(
                "Wrote %g GB at %g GB/s (%g GB/s for distribution, %g GB/s for checksum, %g GB/s for writing, %d views per node)"
                % (
                    szGB,
                    szGB / (t1 - t0),
                    szGB / dt_distr,
                    szGB / dt_crc,
                    szGB / dt_write,
                    len(views_for_node),
                )
            )
        g.globalsum(pos)
        return res + " " + " ".join(["%d" % x for x in pos])

    def read_lattice(self, a):
        g_desc = a[0]
        cv_desc = a[1]
        l_desc = a[2]
        filepos = [int(x) for x in a[3:]]

        # first find grid
        if g_desc not in self.params["grids"]:
            self.params["grids"][g_desc] = gpt.grid_from_description(g_desc)
        g = self.params["grids"][g_desc]

        # create a cartesian view and lattice to load
        l = gpt.lattice(g, l_desc)
        cv0 = gpt.cartesian_view(-1, cv_desc, g.fdimensions, g.cb, l.checkerboard())

        # find tasks for my node
        views_for_node = self.views_for_node(cv0, g)

        # performance
        dt_distr, dt_crc, dt_read = 0.0, 0.0, 0.0
        szGB = 0.0
        g.barrier()
        t0 = gpt.time()

        # need to load all views
        for xk, iview in enumerate(views_for_node):
            g.barrier()
            dt_read -= gpt.time()

            f, pos = self.open_view(
                xk, iview, False, cv_desc, g.fdimensions, g.cb, l.checkerboard()
            )

            cache_key = (
                f"{a[0:3]}_{g.obj}_{xk}_{iview}_{id(pos)}_{l.otype.__name__}_read"
            )
            if cache_key not in self.cache:
                self.cache[cache_key] = {}

            if f is not None:
                f.seek(filepos[iview], 0)
                ntag = int.from_bytes(f.read(4), byteorder="little")
                f.read(ntag)  # not needed if index is present
                crc_exp = int.from_bytes(f.read(4), byteorder="little")
                nd = int.from_bytes(f.read(4), byteorder="little")
                f.read(8 * nd)  # not needed if index is present
                sz = int.from_bytes(f.read(8), byteorder="little")
                data = memoryview(f.read(sz))
                dt_crc -= gpt.time()
                crc_comp = gpt.crc32(data)
                dt_crc += gpt.time()
                assert crc_comp == crc_exp
                sys.stdout.flush()
                szGB += len(data) / 1024.0 ** 3.0
            else:
                assert len(pos) == 0
                data = None

            g.barrier()
            dt_read += gpt.time()
            dt_distr -= gpt.time()
            l[pos, self.cache[cache_key]] = data
            g.barrier()
            dt_distr += gpt.time()

        g.barrier()
        t1 = gpt.time()

        szGB = g.globalsum(szGB)
        if self.verbose and dt_crc != 0.0:
            gpt.message(
                "Read %g GB at %g GB/s (%g GB/s for distribution, %g GB/s for reading + checksum, %g GB/s for checksum, %d views per node)"
                % (
                    szGB,
                    szGB / (t1 - t0),
                    szGB / dt_distr,
                    szGB / dt_read,
                    szGB / dt_crc,
                    len(views_for_node),
                )
            )

        return l

    def write_numpy(self, a):
        if self.glb is not None:
            pos = self.glb.tell()
            buf = io.BytesIO()
            numpy.save(buf, a, allow_pickle=False)
            mv = memoryview(buf.getvalue())
            crc = gpt.crc32(mv)
            self.glb.write(crc.to_bytes(4, byteorder="little"))
            self.glb.write(mv)
            return pos, self.glb.tell()
        return 0, 0

    def read_numpy(self, start, end):
        if gpt.rank() == 0:
            self.glb.seek(start, 0)
            crc32_compare = int.from_bytes(self.glb.read(4), byteorder="little")
            data = self.glb.read(end - start - 4)
        else:
            data = None
            crc32_compare = None
        data = gpt.broadcast(0, data)
        crc32_computed = gpt.crc32(memoryview(data))
        if crc32_compare is not None:
            assert crc32_computed == crc32_compare
        return numpy.load(io.BytesIO(data))

    def write_domain_sparse(self, ctx, sdomain):
        return sdomain.grid.describe()

    def read_domain_sparse(self, sdomain_grid, sdomain_cl):
        def rmnan(x):
            return x[~numpy.isnan(x)[:, 0]]

        local_coordinates = numpy.hstack(
            tuple([rmnan(x[:]).real.astype(numpy.int32) for x in sdomain_cl])
        )
        return gpt.domain.sparse(sdomain_grid, local_coordinates)

    def write(self, objs):
        self.create_index("", objs)
        self.flush()

    def create_index(self, ctx, objs):
        f = self.index_file
        assert f is not None
        if type(objs) == dict:
            f.write("{\n")
            for x in objs:
                f.write(x.encode("unicode_escape").decode("utf-8") + "\n")
                self.create_index("%s/%s" % (ctx, x), objs[x])
            f.write("}\n")
        elif isinstance(
            objs, numpy.ndarray
        ):  # needs to be above list for proper precedence
            f.write("array %d %d\n" % self.write_numpy(objs))
        elif type(objs) == list:
            f.write("[\n")
            for i, x in enumerate(objs):
                self.create_index("%s/%d" % (ctx, i), x)
            f.write("]\n")
        elif type(objs) == tuple:
            f.write("(\n")
            for i, x in enumerate(objs):
                self.create_index("%s/%d" % (ctx, i), x)
            f.write(")\n")
        elif type(objs) == float:
            f.write("float %.16g\n" % objs)
        elif type(objs) == int:
            f.write("int %d\n" % objs)
        elif type(objs) == str:
            f.write("str " + objs.encode("unicode_escape").decode("utf-8") + "\n")
        elif type(objs) == complex:
            f.write("complex %.16g %.16g\n" % (objs.real, objs.imag))
        elif type(objs) == gpt.lattice:
            f.write("lattice %s\n" % self.write_lattice(ctx, objs))
        elif type(objs) == numpy.float64:
            f.write(
                "float %.16g\n" % float(objs)
            )  # improve: avoid implicit type conversion
        elif type(objs) == gpt.grid:
            f.write("grid %s\n" % objs.describe())
        elif type(objs) == gpt.domain.sparse:
            f.write("domain.sparse <\n")
            f.write(self.write_domain_sparse(ctx, objs) + "\n")
            for i, x in enumerate(objs.coordinate_lattices()):
                f.write("lattice %s\n" % self.write_lattice(ctx + "/cl" + str(i), x))
            f.write(">\n")
        else:
            print("Unknown type: ", type(objs))
            assert 0

    def keep_context(self, ctx):
        if self.params["paths"] is None:
            return True
        paths = self.params["paths"]
        if type(paths) == str:
            paths = [paths]
        return sum([1 if fnmatch.fnmatch(ctx, p) else 0 for p in paths]) != 0

    def read_index(self, p, ctx=""):
        cmd = p.cmd()
        if cmd == "{":
            p.skip()
            res = {}
            while True:
                cmd = p.cmd()
                if cmd == "}":
                    p.skip()
                    break
                key = p.get_str(0)
                res[key] = self.read_index(p, ctx + "/" + key)
            return res
        elif cmd == "[":
            p.skip()
            res = []
            while True:
                cmd = p.cmd()
                if cmd == "]":
                    p.skip()
                    break
                res.append(self.read_index(p, ctx + ("/%d" % len(res))))
            return res
        elif cmd == "(":
            p.skip()
            res = []
            while True:
                cmd = p.cmd()
                if cmd == ")":
                    p.skip()
                    break
                res.append(self.read_index(p, ctx + ("/%d" % len(res))))
            return tuple(res)
        elif cmd == "int":
            return int(p.get()[1])
        elif cmd == "float":
            return float(p.get()[1])
        elif cmd == "complex":
            a = p.get()
            return complex(float(a[1]), float(a[2]))
        elif cmd == "str":
            return p.get_str(1)
        elif cmd == "array":
            a = p.get()  # array start end
            if not self.keep_context(ctx):
                return None
            return self.read_numpy(int(a[1]), int(a[2]))
        elif cmd == "lattice":
            a = p.get()
            if not self.keep_context(ctx):
                return None
            return self.read_lattice(a[1:])
        elif cmd == "grid":
            return gpt.grid_from_description(p.get()[1])
        elif cmd == "domain.sparse":
            p.skip()
            a = p.get()
            sdomain_grid = gpt.grid_from_description(a[0])
            assert len(a) == 1
            sdomain_cl = []
            keep = self.keep_context(ctx)
            for i in range(sdomain_grid.nd):
                a = p.get()
                assert a[0] == "lattice"
                sdomain_cl.append(self.read_lattice(a[1:]) if keep else None)
            a = p.get()
            assert a == [">"]
            if not keep:
                return None
            return self.read_domain_sparse(sdomain_grid, sdomain_cl)
        else:
            assert 0


class index_parser:
    def __init__(self, lines):
        self.lines = lines
        self.line = 0

    def peek(self):
        return self.lines[self.line].split(" ")

    def get_str(self, i):
        return (" ".join(self.get()[i:])).encode("utf-8").decode("unicode_escape")

    def cmd(self):
        return self.peek()[0]

    def skip(self):
        self.line += 1

    def get(self):
        r = self.peek()
        self.skip()
        return r

    def eof(self):
        assert self.line < len(self.lines)
        return self.line == (len(self.lines) - 1)


@params_convention()
def writer(filename, params):
    return gpt_io(filename, True, params)


def save(filename, objs, params):

    t0 = gpt.time()

    # create io
    x = gpt_io(filename, True, params)

    # create index
    x.write(objs)

    # close
    x.close()

    # goodbye
    if x.verbose:
        t1 = gpt.time()
        gpt.message("Completed writing %s in %g s" % (filename, t1 - t0))


def load(filename, params):

    # first check if this is right file format
    if not (
        os.path.exists(filename + "/index.crc32")
        and os.path.exists(filename + "/global")
    ):
        raise NotImplementedError()

    # timing
    t0 = gpt.time()

    # create io
    x = gpt_io(filename, False, params)
    if x.verbose:
        gpt.message("Reading %s" % filename)

    # read index
    idx = open(filename + "/index", "rb").read()
    crc_expected = int(open(filename + "/index.crc32", "rt").read(), 16)
    crc_computed = gpt.crc32(memoryview(idx))
    assert crc_expected == crc_computed

    p = index_parser(idx.decode("utf-8", "strict").split("\n"))
    res = x.read_index(p)

    # if multiple chunks are available, return them as a list
    if not p.eof():
        res = [res]
        while not p.eof():
            res.append(x.read_index(p))

    # close
    x.close()

    # goodbye
    if x.verbose:
        t1 = gpt.time()
        gpt.message("Completed reading %s in %g s" % (filename, t1 - t0))

    return res
