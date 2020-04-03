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
import cgpt, gpt, os, io, numpy, sys, fnmatch
from time import time

# get local dir an filename
def get_local_name(root, cv):
    ntotal=cv.ranks
    rank=cv.rank
    dirs=32
    nperdir = ntotal // dirs
    if nperdir < 1:
        nperdir=1
    dirrank=rank//nperdir
    directory = "%s/%2.2d" % (root,dirrank)
    filename="%s/%10.10d.field" % (directory,rank)
    return directory,filename

# gpt io class
class gpt_io:
    def __init__(self, root, params, write):
        self.root = root
        self.params = params
        if not "grids" in self.params:
            self.params["grids"] = {}
        else:
            if type(self.params["grids"]) == gpt.grid:
                self.params["grids"] = [ self.params["grids"] ]
            self.params["grids"] = dict([ (g.describe(),g) for g in self.params["grids"] ])
        self.verbose = gpt.default.is_verbose("io")
        os.makedirs(self.root, exist_ok=True)
        if gpt.rank() == 0:
            if write:
                self.glb = open(root + "/global","wb")
            else:
                self.glb = open(root + "/global","r+b")
        else:
            self.glb = None
        self.loc = {}

    def __del__(self):
        self.close()

    def close(self):
        for x in self.loc:
            self.loc[x].close()
        self.loc={}
        if not self.glb is None:
            self.glb.close()
            self.glb = None

    def write_lattice(self, ctx, l):
        g=l.grid

        # writer configuration
        nwriter=gpt.default.nwriter
        if nwriter > g.Nprocessors:
            nwriter=g.Nprocessors
        ngroups = g.Nprocessors // nwriter

        # create cartesian view for writing
        if "mpi" in self.params:
            cv=gpt.cartesian_view(g.processor,self.params["mpi"],g.gdimensions)

            # make sure that we do not want a cv with more processors than we have in g,
            # this is not yet supported
            assert(cv.ranks <= g.Nprocessors)
        else:
            cv=gpt.cartesian_view(g)

        # directories and files
        if cv.rank < cv.ranks:
            dn,fn=get_local_name(self.root,cv)
            if fn not in self.loc:
                os.makedirs(dn, exist_ok=True)
                self.loc[fn] = open(fn,"wb")
            f = self.loc[fn]
        else:
            f = None

        # description and data
        res=g.describe() + " " + cv.describe() + " " + l.describe()
        t0=time()
        mv=gpt.mview(l[gpt.coordinates(cv)])
        t1=time()
        crc=gpt.crc32(mv)
        t2=time()

        # file positions
        pos=numpy.array([ 0 ] * g.Nprocessors,dtype=numpy.uint64)
        if not f is None:
            pos[g.processor]=f.tell()
        g.globalsum(pos)
        tag=(ctx + "\0").encode("utf-8")
        ntag=len(tag)
        nd=len(l.grid.gdimensions)

        for group in range(ngroups):
            g.barrier()
            szGB=0.0
            tg0=time()
            if g.processor % ngroups == group and not f is None:
                f.write(ntag.to_bytes(4,byteorder='little'))
                f.write(tag)
                f.write(crc.to_bytes(4,byteorder='little'))
                f.write(nd.to_bytes(4,byteorder='little'))
                for i in range(nd):
                    f.write(g.gdimensions[i].to_bytes(4,byteorder='little'))
                for i in range(nd):
                    f.write(( g.gdimensions[i] // g.ldimensions[i]).to_bytes(4,byteorder='little'))
                f.write(len(mv).to_bytes(8,byteorder='little'))
                t3=time()
                f.write(mv)
                f.flush()
                t4=time()
                szGB=len(mv) / 1024.**3.
                if self.verbose:
                    gpt.message("Write %g GB on root node at %g GB /s for distribute, %g GB / s for checksum, %g GB / s for writing" % (szGB,szGB/(t1-t0),szGB/(t2-t1),szGB/(t4-t3)))
            szGB=g.globalsum(szGB)
            tg1=time()
            if self.verbose:
                gpt.message("Globally wrote %g GB in group %d / %d at %g GB / s" % (szGB,group+1,ngroups,szGB/(tg1-tg0)))
        return res + " " + " ".join([ "%d" % x for x in pos[0:cv.ranks] ])

    def read_lattice(self, a):
        g_desc=a[0]
        cv_desc=a[1]
        l_desc=a[2]
        filepos=[ int(x) for x in a[3:] ]

        # first find grid
        if not g_desc in self.params["grids"]:
            self.params["grids"][g_desc]=gpt.grid(g_cesc)
        g=self.params["grids"][g_desc]

        # create a cartesian view and lattice to load
        cv0=gpt.cartesian_view(-1,cv_desc,g.gdimensions)
        l=gpt.lattice(g,l_desc)

        # configure reader
        nreader=gpt.default.nwriter
        if nreader > g.Nprocessors:
            nreader = g.Nprocessors

        # need to load all views
        for iview0 in range(0,cv0.ranks,nreader):
            # do I load the view?
            for iread in range(nreader):
                iview = iview0 + iread
                if iread == g.processor and iview < cv0.ranks:
                    cv=gpt.cartesian_view(iview,cv_desc,g.gdimensions)
                    pos=gpt.coordinates(cv)
                    # read data
                    dn,fn=get_local_name(self.root,cv)
                    f=open(fn,"r+b")
                    f.seek(filepos[iview],0)
                    ntag=int.from_bytes(f.read(4),byteorder='little')
                    f.read(ntag) # not needed if index is present
                    crc_exp=int.from_bytes(f.read(4),byteorder='little')
                    nd=int.from_bytes(f.read(4),byteorder='little')
                    f.read(8*nd) # not needed if index is present
                    sz=int.from_bytes(f.read(8),byteorder='little')
                    data=memoryview(f.read(sz))
                    f.close()
                    crc_comp=gpt.crc32(data)
                    assert(crc_comp == crc_exp)
                    sys.stdout.flush()
                else:
                    pos=gpt.coordinates(cv0) # empty view
                    assert(len(pos) == 0)
                    data=None
                l[pos]=data
            
        return l

    def write_numpy(self, a):
        if not self.glb is None:
            pos=self.glb.tell()
            buf=io.BytesIO()
            numpy.save(buf,a, allow_pickle=False)
            mv=memoryview(buf.getvalue())
            crc=gpt.crc32(mv)
            self.glb.write(crc.to_bytes(4,byteorder='little'))
            self.glb.write(mv)
            return pos,self.glb.tell()
        return 0,0

    def read_numpy(self, start, end):
        if gpt.rank() == 0:
            self.glb.seek(start,0)
            crc32_compare=int.from_bytes(self.glb.read(4),byteorder='little')
            data=self.glb.read(end - start - 4)
        else:
            data=None
            crc32_compare=None
        data=gpt.broadcast(0,data)
        crc32_computed=gpt.crc32(memoryview(data))
        if not crc32_compare is None:
            assert(crc32_computed == crc32_compare)
        return numpy.load(io.BytesIO(data))

    def create_index(self, f, ctx, objs):
        if type(objs) == dict:
            f.write("{\n")
            for x in objs:
                f.write(x.encode("unicode_escape").decode("utf-8") + "\n")
                self.create_index(f,"%s/%s" % (ctx,x),objs[x])
            f.write("}\n")
        elif type(objs) == list:
            f.write("[\n")
            for i,x in enumerate(objs):
                self.create_index(f,"%s/%d" % (ctx,i),x)
            f.write("]\n")
        elif type(objs) == float:
            f.write("float %.16g\n" % objs)
        elif type(objs) == int:
            f.write("int %d\n" % objs)
        elif type(objs) == str:
            f.write("str " + objs.encode("unicode_escape").decode("utf-8") + "\n")
        elif type(objs) == complex:
            f.write("complex %.16g %.16g\n" % (objs.real,objs.imag))
        elif type(objs) == numpy.ndarray:
            f.write("array %d %d\n" % self.write_numpy(objs))
        elif type(objs) == gpt.lattice:
             f.write("lattice %s\n" % self.write_lattice(ctx,objs))
        else:
            assert(0)

    def keep_context(self, ctx):
        if not "paths" in self.params:
            return True
        paths=self.params["paths"]
        if type(paths) == str:
            paths=[ paths ]
        return (sum([ 1 if fnmatch.fnmatch(ctx,p) else 0 for p in paths ]) != 0)

    def read_index(self, p, ctx = ""):
        cmd=p.cmd()
        if cmd == "{":
            p.skip()
            res={}
            while True:
                cmd=p.cmd()
                if cmd == "}":
                    p.skip()
                    break
                key=p.get_str(0)
                res[key]=self.read_index(p,ctx + "/" + key)
            return res
        elif cmd == "[":
            p.skip()
            res=[]
            while True:
                cmd=p.cmd()
                if cmd == "]":
                    p.skip()
                    break
                res.append(self.read_index(p,ctx + ("/%d" % len(res))))
            return res
        elif cmd == "int":
            return int(p.get()[1])
        elif cmd == "float":
            return float(p.get()[1])
        elif cmd == "complex":
            a=p.get()
            return complex(float(a[1]),float(a[2]))
        elif cmd == "str":
            return p.get_str(1)
        elif cmd == "array":
            a=p.get() # array start end
            if not self.keep_context(ctx):
                return None
            return self.read_numpy(int(a[1]),int(a[2]))
        elif cmd == "lattice":
            a=p.get()
            if not self.keep_context(ctx):
                return None
            return self.read_lattice(a[1:])
        else:
            assert(0)


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
        self.line+=1

    def get(self):
        r=self.peek()
        self.skip()
        return r

    
def save(filename, objs, params):

    t0=time()

    # create io
    x=gpt_io(filename,params,True)

    # create index
    f=io.StringIO("")
    x.create_index(f,"",objs)
    mvidx=memoryview(f.getvalue().encode("utf-8"))

    # write index to fs
    index_crc=gpt.crc32(mvidx)
    if gpt.rank() == 0:
        open(filename + "/index","wb").write(mvidx)
        open(filename + "/index.crc32","wt").write("%X\n" % index_crc)

    # close
    x.close()

    # goodbye
    if x.verbose:
        t1=time()
        gpt.message("Completed writing %s in %g s" % (filename,t1-t0))


def load(filename, *a):

    # first check if this is right file format
    if not os.path.exists(filename + "/index.crc32"):
        raise NotImplementedError()

    # parameters
    if len(a) == 0:
        params={}
    else:
        params=a[0]

    # timing
    t0=time()

    # create io
    x=gpt_io(filename,params,False)

    # read index
    idx=open(filename + "/index","rb").read()
    crc_expected=int(open(filename + "/index.crc32","rt").read(),16)
    crc_computed=gpt.crc32(memoryview(idx))
    assert(crc_expected == crc_computed)

    p=index_parser(idx.decode("utf-8","strict").split("\n"))
    res=x.read_index(p)

    # close
    x.close()

    # goodbye
    if x.verbose:
        t1=time()
        gpt.message("Completed reading %s in %g s" % (filename,t1-t0))

    return res
