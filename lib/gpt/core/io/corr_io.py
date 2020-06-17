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
import sys, os, struct, binascii, fnmatch, numpy, gpt

def read_tags(fn, tags, verbose, nocheck):
    f=open(fn,"r+b")
    try:
        while True:
            rd=f.read(4)
            if len(rd) == 0:
                break
            ntag=struct.unpack('i', rd)[0]
            tag=f.read(ntag)
            (crc32,ln)=struct.unpack('II', f.read(4*2))

            if verbose:
                gpt.message(tag)

            data=f.read(16*ln)
            if nocheck == False:
                crc32comp= ( binascii.crc32(data) & 0xffffffff)

                if crc32comp != crc32:
                    raise Exception("Data corrupted!")

            cdata=numpy.frombuffer(data,dtype=numpy.complex128,count=ln)
            if nocheck == False:
                cdata.tolist()
            tags[tag[0:-1]]=cdata
        f.close()
    except:
        raise

def write_tag(f,t,cc):
    f.write(struct.pack('i', len(t)+1))
    f.write((t + "\0").encode("utf-8"))
    ln=len(cc)
    ccr=[fff for sublist in ((c.real, c.imag) for c in cc) for fff in sublist]
    bindata=struct.pack('d'*2*ln,*ccr)
    crc32comp= ( binascii.crc32(bindata) & 0xffffffff)
    f.write(struct.pack('II', crc32comp,ln))
    f.write(bindata)

def write_tags(fn, tags):
    f=open(fn,"w+b")
    try:
        for t in tags:
            cc=tags[t]
            write_tag(f,t,cc)
        f.close()
    except:
        raise Exception()

class corr_io:

    def __init__(self, fn, mode = "r", verbose = False, nocheck = False):
        self.tags = {}

        if mode == "r":
            if type(fn) == type(""):
                read_tags(fn, self.tags,verbose, nocheck)
            elif type(fn) == type([]):
                for f in fn:
                    read_tags(f, self.tags,verbose, nocheck)
            else:
                raise Exception("Unknown argument type")
        elif mode == "w":
            self.fn = fn
        else:
            raise Exception("Unknown mode")
            

    def glob(self, pattern):
        return filter(lambda k: fnmatch.fnmatch(k,pattern),self.tags.keys())

    def keys_match_post(self, pf):
        l=len(pf)
        return [ k[0:-l] for k in self.tags.keys() if k[-l:]==pf ]

    def write(self):
        write_tags(self.fn, self.tags)

class writer:
    def __init__(self, fn):
        self.f=open(fn,"w+b")

    def write(self, tag, cc):
        write_tag(self.f,tag, cc)
        self.f.flush()

    def __del__(self):
        self.f.close()
