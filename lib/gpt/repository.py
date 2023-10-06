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
import sys
import os
import shutil
from urllib import request

base = {"gpt:": "https://raw.githubusercontent.com/lehner/gpt-repository/master"}


def download(dst, src):
    src_a = src.split("/")
    assert len(src_a) >= 3

    host = src_a[0]
    assert host[-1] == ":"

    assert src_a[1] == ""

    path = "/".join(src_a[2:])

    assert host in base

    baseurl = base[host]

    if gpt.rank() == 0:
        verbose = gpt.default.is_verbose("repository")
        t0 = gpt.time()
        if "GPT_REPOSITORY" in os.environ:
            root = os.environ["GPT_REPOSITORY"]
            shutil.copy2(f"{root}/{path}", dst)
            mode = "copy"
        else:
            filename, header = request.urlretrieve(f"{baseurl}/{path}", filename=dst)
            mode = "download"
        t1 = gpt.time()
        filesize = os.path.getsize(dst)
        speedMBs = filesize / 1024.0**2.0 / (t1 - t0)
        if verbose:
            gpt.message(f"Repository {mode} {src} in {t1-t0:g} s at {speedMBs:g} MB/s")

    # add a barrier so that all nodes have file after download
    gpt.barrier()

    # os.scandir to trigger network filesystem synchronization
    os.scandir(os.path.dirname(dst))


class repository:
    def load(first, second=None):
        # params
        if second is None:
            src = first
            dst = first.split("/")[-1]
        else:
            src = second
            dst = first

        # if dst already exists, return
        if os.path.exists(dst):
            return dst

        # download from repository
        download(dst, src)

        # succeess
        return dst
