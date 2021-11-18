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

# , os, shutil, sys

# def cache_file(src,md):
#     if md != "rb":
#         return src
#     root="/scratch"
#     dst = "%s/%s" % (root,src.replace("/","_"))
#     src_size = os.stat(src).st_size
#     print(f"Caching {src} of size {src_size} to {dst}"); sys.stdout.flush()
#     if os.path.exists(dst):
#         if os.stat(dst).st_size == src_size:
#             print("Use cached"); sys.stdout.flush()
#             return dst
#         else:
#             os.unlink(dst)
#     print("Start copy"); sys.stdout.flush()
#     shutil.copyfile(src,dst)
#     print("End copy"); sys.stdout.flush()
#     return dst


class FILE:
    def __init__(self, fn, md):
        # fn = cache_file(fn,md)
        self.f = cgpt.fopen(fn, md)
        if self.f == 0:
            self.f = None
            raise FileNotFoundError("Can not open file %s" % fn)

    def __del__(self):
        if self.f is not None:
            cgpt.fclose(self.f)

    def close(self):
        assert self.f is not None
        cgpt.fclose(self.f)
        self.f = None

    def tell(self):
        assert self.f is not None
        r = cgpt.ftell(self.f)
        return r

    def seek(self, offset, whence):
        assert self.f is not None
        r = cgpt.fseek(self.f, offset, whence)
        return r

    def read(self, sz):
        assert self.f is not None
        t = bytes(sz)
        if sz > 0:
            if cgpt.fread(self.f, sz, memoryview(t)) != 1:
                t = bytes(0)
        return t

    def write(self, d):
        assert self.f is not None
        if type(d) != memoryview:
            d = memoryview(d)
        assert cgpt.fwrite(self.f, len(d), d) == 1

    def flush(self):
        assert self.f is not None
        cgpt.fflush(self.f)
