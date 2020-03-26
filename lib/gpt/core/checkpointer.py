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
        directory = "%s/%2.2d" % (root, gpt.rank() // 32 )
        os.makedirs(directory, exist_ok=True)
        self.filename = "%s/%10.10d" % (directory,gpt.rank())
        try:
            self.f = open(self.filename,"r+b")
        except:
            self.f = open(self.filename,"w+b")
        self.f.seek(0,1)
        self.verbose = gpt.default.is_verbose("checkpointer")

    def save(self, obj):
        if type(obj) == list:
            for o in obj:
                self.save(o)
        elif type(obj) == gpt.lattice:
            self.save(obj.mview())
        elif type(obj) == float:
            self.save(memoryview(struct.pack("d",obj)))
        elif type(obj) == complex:
            self.save(memoryview(struct.pack("dd",obj.real,obj.imag)))
        elif type(obj) == memoryview:
            self.f.seek(0,1)
            sz=len(obj)
            szGB=sz/1024.**3
            self.f.write(sz.to_bytes(8,'little'))
            t0=gpt.time()
            self.f.write(gpt.crc32(obj).to_bytes(4,'little'))
            t1=gpt.time()
            self.f.write(obj)
            self.f.flush()
            t2=gpt.time()
            if self.verbose:
                gpt.message("Checkpoint %g GB on head node at %g GB/s for crc32 and %g GB/s for write in %g s total" % (szGB,szGB/(t1-t0),szGB/(t2-t1),t2-t0))
        else:
            assert(0)

    def load(self, obj):
        if type(obj) == list:
            if len(obj) != 1:
                allok=True
                pos=self.f.tell()
                for i,o in enumerate(obj):
                    r=[o]
                    allok=allok and self.load(r)
                    obj[i]=r[0]
                if not allok:
                    self.f.seek(pos,0) # reset position to overwrite corrupted data chunk
                return allok
            else:
                if type(obj[0]) == gpt.lattice:
                    res=self.load(obj[0].mview())
                elif type(obj[0]) == float:
                    v=memoryview(bytearray(8))
                    res=self.load(v)
                    obj[0]=struct.unpack("d",v)[0]
                elif type(obj) == complex:
                    v=memoryview(bytearray(16))
                    res=self.load(v)
                    obj[0]=complex(*struct.unpack("dd",v)[0,1])
                else:
                    assert(0)
                return res
        elif type(obj) == memoryview:
            return self.read_view(obj)
        else:
            assert(0)

    def read_view(self, obj):
        pos=self.f.tell()
        self.f.seek(0,2)
        flags=numpy.array([0.0,1.0],dtype=numpy.float64)
        if self.f.tell() != pos:
            self.f.seek(pos,0)
                
            # try to read
            sz=int.from_bytes(self.f.read(8),'little')
            szGB=sz/1024.**3
            crc32_expected=int.from_bytes(self.f.read(4),'little')
            if len(obj) == sz:
                t0=gpt.time()
                data=self.f.read(sz)
                t1=gpt.time()
                if len(data) == sz:
                    obj[:]=data
                    crc32=gpt.crc32(obj)
                    t2=gpt.time()
                    if crc32 == crc32_expected:
                        flags[0]=1.0 # flag success on this node

            # compare global
            assert(not self.grid is None)
            self.grid.globalsum(flags)

            # report status
            if self.verbose:
                if flags[0] != flags[1]:
                    gpt.message("Checkpoint %g GB per node read failed on %g out of %g nodes",szGB,flags[0],flags[1])
                else:
                    gpt.message("Checkpoint %g GB on head node at %g GB/s for crc32 and %g GB/s for read in %g s total" % (szGB,szGB/(t2-t1),szGB/(t1-t0),t2-t0))

            # all nodes OK?
            if flags[0] == flags[1]:
                return True
                
            # reset position to overwrite corruption
            self.f.seek(pos,0)
        
        return False
