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
import cgpt, gpt, os, io, numpy
from time import time

# get local dir an filename
def get_local_name(root, grid):
    ntotal=grid.Nprocessors
    rank=grid.processor
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
    def __init__(self, root):
        self.root = root
        self.verbose = gpt.default.is_verbose("io")
        os.makedirs(self.root, exist_ok=True)
        if gpt.rank() == 0:
            self.glb = open(root + "/global","wb")
        else:
            self.glb = None
        self.loc = None

    def __del__(self):
        self.close()

    def close(self):
        if not self.loc is None:
            self.loc.close()
            self.loc = None
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

        # directories and files
        dn,fn=get_local_name(self.root,g)
        if self.loc is None:
            os.makedirs(dn, exist_ok=True)
            self.loc = open(fn,"wb")

        # description and data
        res=g.describe() + " " + l.describe()
        t0=time()
        data=cgpt.mview(l[gpt.coordinates(g)])
        t1=time()
        crc=gpt.crc32(data)
        t2=time()

        # file positions
        pos=numpy.array([ 0 ] * g.Nprocessors,dtype=numpy.uint64)
        pos[g.processor]=self.loc.tell()
        g.globalsum(pos)
        tag=(ctx + "\0").encode("utf-8")
        ntag=len(tag)
        nd=len(l.grid.gdimensions)

        for group in range(ngroups):
            g.barrier()
            tg0=time()
            if g.processor % ngroups == group:
                self.loc.write(ntag.to_bytes(4,byteorder='little'))
                self.loc.write(tag)
                self.loc.write(crc.to_bytes(4,byteorder='little'))
                self.loc.write(nd.to_bytes(4,byteorder='little'))
                for i in range(nd):
                    self.loc.write(g.gdimensions[i].to_bytes(4,byteorder='little'))
                for i in range(nd):
                    self.loc.write(( g.gdimensions[i] // g.ldimensions[i]).to_bytes(4,byteorder='little'))
                self.loc.write(len(data).to_bytes(8,byteorder='little'))
                t3=time()
                self.loc.write(data)
                self.loc.flush()
                t4=time()
                szGB=len(data) / 1024.**3.
                if self.verbose:
                    gpt.message("Write %g GB on root node at %g GB /s for distribute, %g GB / s for checksum, %g GB / s for writing" % (szGB,szGB/(t1-t0),szGB/(t2-t1),szGB/(t4-t3)))
            else:
                szGB=0.0
            szGB=g.globalsum(szGB)
            tg1=time()
            if self.verbose:
                gpt.message("Globally wrote %g GB in group %d / %d at %g GB / s" % (szGB,group+1,ngroups,szGB/(tg1-tg0)))
        return res + " " + " ".join([ "%d" % x for x in pos ])

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


def save(filename, objs):
    
    # create io
    x=gpt_io(filename)

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
    
