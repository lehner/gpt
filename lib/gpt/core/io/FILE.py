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
import cgpt, gpt, os, sys, shutil, zipfile, time


def cache_file(root, src, md):
    if md != "rb":
        return src
    dst = "%s/%s" % (root, src.replace("/", "_"))
    src_size = os.stat(src).st_size
    if os.path.exists(dst):
        if os.stat(dst).st_size == src_size:
            return dst
        else:
            os.unlink(dst)
    shutil.copyfile(src, dst)
    return dst


cache_root = gpt.default.get("--cache-root", None)


def zip_split(fn):
    # check if any of the directories is a .zip archive
    fn_split = fn.split("/")
    fn_partial = fn_split[0]
    for i, fn_part in enumerate(fn_split[1:]):
        zip_candidate = fn_partial + ".zip"
        if os.path.exists(zip_candidate):
            return zip_candidate, "/".join(fn_split[i:])
        fn_partial = f"{fn_partial}/{fn_part}"
    return None, None


def FILE_exists(fn):
    if os.path.exists(fn):
        return True

    fn_zip, fn_element = zip_split(fn)
    if fn_zip is not None:
        return True

    return False


class FILE_base:
    def __init__(self, fn, md):
        # do not support appending
        if cache_root is not None:
            fn = cache_file(cache_root, fn, md)
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
        return cgpt.ftell(self.f)

    def seek(self, offset, whence):
        assert self.f is not None
        assert cgpt.fseek(self.f, offset, whence) == 0

    def read(self, sz=None):
        if sz is None:
            pos = self.tell()
            self.seek(0, 2)
            size = self.tell()
            self.seek(pos, 0)
            return self.read(size - pos)

        assert self.f is not None
        try:
            t = bytes(sz)
        except OverflowError:
            raise IOError(f"Cannot allocate {sz} bytes")
        if sz > 0:
            if cgpt.fread(self.f, sz, memoryview(t)) != 1:
                t = bytes(0)
        return t

    def write(self, d):
        assert self.f is not None
        if not isinstance(d, memoryview):
            d = memoryview(d)
        assert cgpt.fwrite(self.f, len(d), d) == 1


class FILE_windowed_reader:
    def __init__(self, f, offset, size):
        self.f = FILE_base(f, "rb")
        self.offset = offset
        self.pos = 0
        self.size = size
        self.f.seek(self.offset, 0)

    def read(self, sz=None):
        left = self.size - self.pos
        assert left >= 0
        if sz is None:
            sz = left
        if sz > left:
            sz = left
        data = self.f.read(sz)
        self.pos += sz
        return data

    def seek(self, offset, whence):
        if whence == 0:
            if offset < 0:
                offset = 0
            if offset >= self.size:
                offset = self.size
            self.f.seek(self.offset + offset, 0)
            self.pos = offset
        else:
            raise Exception("Not yet implemented")

    def close(self):
        self.f.close()


def FILE(fn, mode):
    if mode[0] != "r" or "+" in mode or os.path.exists(fn):
        return FILE_base(fn, mode)

    # if file does not exists but should be read, try zip route
    fn_zip, fn_element = zip_split(fn)
    zip_file = zipfile.ZipFile(fn_zip, "r")
    f = zip_file.open(fn_element, "r")
    if f._compress_type != zipfile.ZIP_STORED:
        # slow random access (seek), however, it may be useful if we do not plan to randomly access files
        return f

    # return my own reader that has O(1) seek cost, currently zipfile does not implement such O(1) seeks
    offset = f._orig_compress_start
    compress_size = f._orig_compress_size
    file_size = f._orig_file_size
    assert compress_size == file_size
    return FILE_windowed_reader(fn_zip, offset, file_size)
