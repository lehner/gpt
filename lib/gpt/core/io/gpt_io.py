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
import cgpt, gpt, os, io, numpy, sys
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
    def __init__(self, root, params):
        self.root = root
        self.params = params
        self.verbose = gpt.default.is_verbose("io")
        os.makedirs(self.root, exist_ok=True)
        if gpt.rank() == 0:
            self.glb = open(root + "/global","wb")
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
        data=cgpt.mview(l[gpt.coordinates(cv)])
        t1=time()
        crc=gpt.crc32(data)
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
                f.write(len(data).to_bytes(8,byteorder='little'))
                t3=time()
                f.write(data)
                f.flush()
                t4=time()
                szGB=len(data) / 1024.**3.
                if self.verbose:
                    gpt.message("Write %g GB on root node at %g GB /s for distribute, %g GB / s for checksum, %g GB / s for writing" % (szGB,szGB/(t1-t0),szGB/(t2-t1),szGB/(t4-t3)))
            szGB=g.globalsum(szGB)
            tg1=time()
            if self.verbose:
                gpt.message("Globally wrote %g GB in group %d / %d at %g GB / s" % (szGB,group+1,ngroups,szGB/(tg1-tg0)))
        return res + " " + " ".join([ "%d" % x for x in pos[0:cv.ranks] ])

    def write_numpy(self, a):
        if not self.glb is None:
            pos=self.glb.tell()
            buf=io.BytesIO()
            numpy.save(buf,a)
            mv=memoryview(buf.getvalue())
            crc=gpt.crc32(mv)
            self.glb.write(crc.to_bytes(4,byteorder='little'))
            self.glb.write(mv)
            return pos,self.glb.tell()
        return 0,0

    def create_index(self, f, ctx, objs):
        if type(objs) == dict:
            f.write("{\n")
            for x in objs:
                f.write("%s\n" % repr(x))
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
            f.write("str %s\n" % repr(objs))
        elif type(objs) == complex:
            f.write("complex %.16g %.16g\n" % (objs.real,objs.imag))
        elif type(objs) == numpy.ndarray:
            f.write("array %d %d\n" % self.write_numpy(objs))
        elif type(objs) == gpt.lattice:
             f.write("lattice %s\n" % self.write_lattice(ctx,objs))
        else:
            assert(0)


def save(filename, objs, params):

    t0=time()

    # create io
    x=gpt_io(filename,params)

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

