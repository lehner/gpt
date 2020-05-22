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
import cgpt, gpt

class FILE:
    def __init__(self, fn, md):
        t0=gpt.time()
        self.f = cgpt.fopen(fn,md)
        if self.f == 0:
            self.f = None
            raise FileNotFoundError("Can not open file %s" % fn)
        t1=gpt.time()
        #print("OPEN %g" % (t1-t0))

    def __del__(self):
        t0=gpt.time()
        if not self.f is None:
            cgpt.fclose(self.f)
        t1=gpt.time()
        #print("__del__ %g" % (t1-t0))

    def close(self):
        t0=gpt.time()
        assert(not self.f is None)
        cgpt.fclose(self.f)
        self.f = None
        t1=gpt.time()
        #print("CLOSE %g" % (t1-t0))

    def tell(self):
        assert(not self.f is None)
        t0=gpt.time()
        r=cgpt.ftell(self.f)
        t1=gpt.time()
        #print("TELL %g" % (t1-t0))
        return r

    def seek(self, offset, whence):
        assert(not self.f is None)
        t0=gpt.time()
        r=cgpt.fseek(self.f, offset, whence)
        t1=gpt.time()
        #print("SEEK %g" % (t1-t0))
        return r

    def read(self, sz):
        assert(not self.f is None)
        t0=gpt.time()
        t=bytes(sz)
        if sz > 0:
            assert(cgpt.fread(self.f,sz,memoryview(t))==1)
        t1=gpt.time()
        #print("READ %g s, %g GB" % (t1-t0,sz/1024.**3.))
        return t

    def write(self, d):
        assert(not self.f is None)
        t0=gpt.time()
        if type(d) != memoryview:
            d=memoryview(d)
        assert(cgpt.fwrite(self.f,len(d),d)==1)
        t1=gpt.time()
        #print("WRITE %g s, %g GB" % (t1-t0,len(d)/1024.**3.))

    def flush(self):
        assert(not self.f is None)
        t0=gpt.time()
        cgpt.fflush(self.f)
        t1=gpt.time()
        #print("FLUSH %g" % (t1-t0))
