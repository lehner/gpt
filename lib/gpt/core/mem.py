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
import resource, gpt, cgpt, os


class accelerator:
    pass


class host:
    pass


def mem_host_available():
    try:
        res = dict(
            [
                ln.split(":")
                for ln in filter(lambda x: x != "", open("/proc/meminfo").read().split("\n"))
            ]
        )
        return float(res["MemAvailable"].strip().split(" ")[0]) * 1024.0
    except Exception:
        return float("nan")


def mem_info():
    return {
        **{
            "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024.0,
            "host_available": mem_host_available(),
        },
        **cgpt.util_mem(),
    }


def mem_report(details=True):
    info = mem_info()
    mem_book = gpt.get_mem_book()
    fmt = " %-8s %-30s %-12s %-30s %-12s %-16s %-20s"
    gpt.message(
        "===================================================================================================================================="
    )
    gpt.message(
        "                                                 GPT Memory Report                "
    )
    g_tot_gb = 0.0
    l_tot_gb = 0.0
    if len(mem_book) > 0:
        if details:
            gpt.message(
                "===================================================================================================================================="
            )
            gpt.message(
                fmt
                % (
                    "Index",
                    "Grid",
                    "Precision",
                    "OType",
                    "CBType",
                    "Size/GB",
                    "Created at time",
                )
            )
        smsg_prev = ""
        for i, page in enumerate(mem_book):
            grid, otype, created, stack = mem_book[page]
            g_gb = grid.fsites * grid.precision.nbytes * otype.nfloats / grid.cb.n / 1024.0**3.0
            l_gb = g_gb / grid.Nprocessors
            g_tot_gb += g_gb
            l_tot_gb += l_gb
            if details:
                if stack is not None:
                    sfmt = "\n" + (" " * 10) + f"%-{73}s  %s"
                    smsg = ""
                    for iline, line in enumerate(stack):
                        smsg += sfmt % (
                            (" " * iline) + os.path.basename(line[0]),
                            line[1].strip(),
                        )
                    if smsg != smsg_prev:
                        smsg_prev = smsg
                        gpt.message(
                            "------------------------------------------------------------------------------------------------------------------------------------"
                        )
                        gpt.message(smsg + "\n")
                gpt.message(
                    fmt
                    % (
                        i,
                        grid.gdimensions,
                        grid.precision.__name__,
                        otype.__name__,
                        grid.cb.__name__,
                        "%g" % g_gb,
                        "%.6f s" % created,
                    )
                )
    gpt.message(
        "===================================================================================================================================="
    )
    gpt.message(" %-39s %g GB" % ("Lattice fields on all ranks", g_tot_gb))
    gpt.message(" %-39s %g GB" % ("Lattice fields per rank", l_tot_gb))
    gpt.message(" %-39s %g GB" % ("Resident memory per rank", info["maxrss"] / 1024**3.0))
    gpt.message(
        " %-39s %g GB" % ("Total memory available (host)", info["host_available"] / 1024**3.0)
    )
    gpt.message(
        " %-39s %g GB"
        % (
            "Total memory available (accelerator)",
            info["accelerator_available"] / 1024**3.0,
        )
    )
    gpt.message(
        "===================================================================================================================================="
    )
