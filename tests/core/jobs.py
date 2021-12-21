#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2021
#
import os, sys, shutil
import gpt as g


class job_create_file(g.jobs.base):
    def __init__(self, fn, needfn):
        self.fn = fn
        super().__init__("file_" + fn, ["file_" + f for f in needfn])

    def perform(self, root):
        f = open(f"{root}/{self.name}/{self.fn}", "wt")
        f.write(self.fn + "\n")
        f.close()

    def check(self, root):
        return os.path.exists(f"{root}/{self.name}/{self.fn}")


if os.path.exists("test.root") and g.rank() == 0:
    shutil.rmtree("test.root")

g.barrier()

fail_B = True

for jid in range(5):
    j = g.jobs.next(
        "test.root",
        [
            job_create_file("A", []),
            job_create_file("C", ["B", "A"]),
            job_create_file("B", ["A"]),
        ],
        stale_seconds=0.0,
    )

    if j is not None and j.name == "file_B" and fail_B:
        if g.rank() == 0:
            os.unlink("test.root/file_B/.completed")
            g.message("Fail B")
        g.barrier()
        fail_B = False
