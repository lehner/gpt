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
import os, time, sys
import gpt as g


class base:
    def __init__(self, name, needs):
        self.name = name
        self.needs = needs
        self.weight = 1.0

    def perform(self, root):
        raise NotImplementedError(f"{self.name} perform not implemented")

    def check(self, root):
        raise NotImplementedError(f"{self.name} perform not implemented")

    def has_started(self, root):
        return os.path.exists(f"{root}/{self.name}")

    def reserve_start(self, root):
        os.makedirs(f"{root}/{self.name}", exist_ok=True)

    def has_completed(self, root):
        fd = f"{root}/{self.name}"
        if os.path.exists(f"{fd}/.checked"):
            return True
        if os.path.exists(f"{fd}/.completed"):
            if self.check(root):
                f = open(f"{fd}/.checked", "wt")
                f.write(time.asctime() + "\n")
                f.close()
                return True
        return False

    def __call__(self, root):
        fd = f"{root}/{self.name}"
        os.makedirs(fd, exist_ok=True)

        f = open(f"{fd}/.started", "wt")
        f.write(time.asctime() + "\n")
        f.write(str(sys.argv) + "\n")
        f.write(str(os.environ) + "\n")
        f.close()

        self.perform(root)

        f = open(f"{fd}/.completed", "wt")
        f.write(time.asctime() + "\n")
        f.close()


def get_next_name(root, jobs):
    # create lut
    lut = {}
    for j in jobs:
        lut[j.name] = j

    for j in jobs:
        if not j.has_started(root):
            # check dependencies
            dependencies_ok = True
            for dep_j in [lut[d] for d in j.needs]:
                if not dep_j.has_completed(root):
                    dependencies_ok = False
                    g.message(
                        f"Dependency {dep_j.name} of {j.name} is not yet satisfied."
                    )
                    break
            if dependencies_ok:
                # last check if in meantime somebody else has started running same job
                if not j.has_started(root):
                    j.reserve_start(root)
                    return j.name

    return ""


def next(root, jobs):
    if g.rank() == 0:
        j = get_next_name(root, jobs).encode("utf-8")
    else:
        j = bytes()

    j_name = g.broadcast(0, j).decode("utf-8")
    for j in jobs:
        if j.name == j_name:
            g.message(
                f"""
--------------------------------------------------------------------------------
   Start job {j.name}
--------------------------------------------------------------------------------
"""
            )
            t0 = g.time()
            j(root)
            t1 = g.time()
            g.message(
                f"""
--------------------------------------------------------------------------------
   Completed {j.name} in {t1-t0} seconds
--------------------------------------------------------------------------------
"""
            )
            return j
    return None
