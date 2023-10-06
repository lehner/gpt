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
#    Format of
#
#       https://github.com/lehner/eigen-comp
#
#    used in the original implementation of
#
#       https://arxiv.org/abs/1710.06884 .
#
import cgpt, gpt, os, io, numpy, sys
from gpt.params import params_convention


# get local dir an filename
def get_local_name(root, cv):
    if cv.rank < 0:
        return None, None
    ntotal = cv.ranks
    rank = cv.rank
    dirs = 32
    nperdir = ntotal // dirs
    if nperdir < 1:
        nperdir = 1
    dirrank = rank // nperdir
    directory = "%s/%2.2d" % (root, dirrank)
    filename = "%s/%10.10d.compressed" % (directory, rank)
    return directory, filename


def get_param(params, a, v):
    if a in params:
        return params[a]
    return v


def mem_avail():
    return gpt.mem_info()["host_available"] / 1024**3.0


def conformDiv(a, b):
    assert a % b == 0
    return a // b


def FP_16_SIZE(a, b):
    assert a % b == 0
    return ((a) + (a // b)) * 2


def read_metadata(fn):
    return dict(
        [
            tuple([x.strip() for x in ln.split("=")])
            for ln in filter(lambda x: x != "" and x[0] != "#", open(fn).read().split("\n"))
        ]
    )


def get_vec(d, n, conv):
    i = 0
    r = []
    while True:
        tag = "%s[%d]" % (n, i)
        i += 1
        if tag in d:
            r.append(conv(d[tag]))
        else:
            return r


def get_ivec(d, n):
    return get_vec(d, n, int)


def get_xvec(d, n):
    return get_vec(d, n, lambda x: int(x, 16))


@params_convention(grids=None, nmax=None, advise_basis=None, advise_cevec=None)
def load(filename, params):
    # first check if this is right file format
    if not os.path.exists(filename + "/00/0000000000.compressed") or not os.path.exists(
        filename + "/metadata.txt"
    ):
        raise NotImplementedError()

    # verbosity
    verbose = gpt.default.is_verbose("io")

    # site checkerboard
    # only odd is used in this file format but
    # would be easy to generalize here
    site_cb = gpt.odd

    # need grids parameter
    assert params["grids"] is not None
    assert isinstance(params["grids"], gpt.grid)
    fgrid = params["grids"]
    assert fgrid.precision == gpt.single
    fdimensions = fgrid.fdimensions

    # read metadata
    metadata = read_metadata(filename + "/metadata.txt")
    s = get_ivec(metadata, "s")
    ldimensions = [s[4]] + s[:4]
    blocksize = get_ivec(metadata, "b")
    blocksize = [blocksize[4]] + blocksize[:4]
    nb = get_ivec(metadata, "nb")
    nb = [nb[4]] + nb[:4]
    crc32 = get_xvec(metadata, "crc32")
    neigen = int(metadata["neig"])
    nbasis = int(metadata["nkeep"])
    nsingle = int(metadata["nkeep_single"])
    blocks = int(metadata["blocks"])
    FP16_COEF_EXP_SHARE_FLOATS = int(metadata["FP16_COEF_EXP_SHARE_FLOATS"])
    nsingleCap = min([nsingle, nbasis])

    # check
    nd = len(ldimensions)
    assert nd == 5
    assert nd == len(fdimensions)
    assert nd == len(blocksize)
    assert fgrid.cb.n == 2
    assert fgrid.cb.cb_mask == [0, 1, 1, 1, 1]

    # create coarse grid
    cgrid = gpt.block.grid(fgrid, blocksize)

    # allow for partial loading of data
    if params["nmax"] is not None:
        nmax = params["nmax"]
        nbasis_max = min([nmax, nbasis])
        neigen_max = min([nmax, neigen])
        nsingleCap_max = min([nmax, nsingleCap])
    else:
        nbasis_max = nbasis
        neigen_max = neigen
        nsingleCap_max = nsingleCap

    # allocate all lattices
    basis = [gpt.vspincolor(fgrid) for i in range(nbasis_max)]
    cevec = [gpt.vcomplex(cgrid, nbasis) for i in range(neigen_max)]
    if params["advise_basis"] is not None:
        gpt.advise(basis, params["advise_basis"])
    if params["advise_cevec"] is not None:
        gpt.advise(cevec, params["advise_cevec"])

    # fix checkerboard of basis
    for i in range(nbasis_max):
        basis[i].checkerboard(site_cb)

    # mpi layout
    mpi = []
    for i in range(nd):
        assert fdimensions[i] % ldimensions[i] == 0
        mpi.append(fdimensions[i] // ldimensions[i])
    assert mpi[0] == 1  # assert no mpi in 5th direction

    # create cartesian view on fine grid
    cv0 = gpt.cartesian_view(-1, mpi, fdimensions, fgrid.cb, site_cb)
    views = cv0.views_for_node(fgrid)

    # timing
    totalSizeGB = 0
    dt_fp16 = 1e-30
    dt_distr = 1e-30
    dt_munge = 1e-30
    dt_crc = 1e-30
    dt_fread = 1e-30
    t0 = gpt.time()

    # load all views
    if verbose:
        gpt.message("Loading %s with %d views per node" % (filename, len(views)))
    for i, v in enumerate(views):
        cv = gpt.cartesian_view(v if v is not None else -1, mpi, fdimensions, fgrid.cb, site_cb)
        cvc = gpt.cartesian_view(
            v if v is not None else -1, mpi, cgrid.fdimensions, gpt.full, gpt.none
        )
        pos_coarse = gpt.coordinates(cvc, "canonical")

        dn, fn = get_local_name(filename, cv)

        # sizes
        slot_lsites = numpy.prod(cv.view_dimensions)
        assert slot_lsites % blocks == 0
        block_data_size_single = slot_lsites * 12 // 2 // blocks * 2 * 4
        block_data_size_fp16 = FP_16_SIZE(slot_lsites * 12 // 2 // blocks * 2, 24)
        coarse_block_size_part_fp32 = 2 * (4 * nsingleCap)
        coarse_block_size_part_fp16 = 2 * (
            FP_16_SIZE(nbasis - nsingleCap, FP16_COEF_EXP_SHARE_FLOATS)
        )
        coarse_vector_size = (coarse_block_size_part_fp32 + coarse_block_size_part_fp16) * blocks
        coarse_fp32_vector_size = 2 * (4 * nbasis) * blocks

        # checksum
        crc32_comp = 0

        # file
        f = gpt.FILE(fn, "rb") if fn is not None else None

        # block positions
        pos = [
            cgpt.coordinates_from_block(cv.top, cv.bottom, b, nb, "canonicalOdd")
            for b in range(blocks)
        ]

        # group blocks
        read_blocks = blocks
        block_reduce = 1
        max_read_blocks = get_param(params, "max_read_blocks", 8)
        for divisor in [2, 3, 5]:
            while read_blocks > max_read_blocks and read_blocks % divisor == 0:
                pos = [
                    numpy.concatenate(tuple([pos[divisor * i + j] for j in range(divisor)]))
                    for i in range(read_blocks // divisor)
                ]
                block_data_size_single *= divisor
                block_data_size_fp16 *= divisor
                read_blocks //= divisor
                block_reduce *= divisor
        if verbose:
            gpt.message("Read blocks", read_blocks)

        # make read-only to enable caching
        for x in pos:
            x.setflags(write=0)

        # dummy buffer
        data0 = memoryview(bytes())

        # single-precision data
        data_munged = memoryview(bytearray(block_data_size_single * nsingleCap))
        for b in range(read_blocks):
            fgrid.barrier()
            dt_fread -= gpt.time()
            if f is not None:
                data = memoryview(f.read(block_data_size_single * nsingleCap))
                globalReadGB = len(data) / 1024.0**3.0
            else:
                globalReadGB = 0.0
            globalReadGB = fgrid.globalsum(globalReadGB)
            dt_fread += gpt.time()
            totalSizeGB += globalReadGB

            if f is not None:
                dt_crc -= gpt.time()
                crc32_comp = gpt.crc32(data, crc32_comp)
                dt_crc += gpt.time()
                dt_munge -= gpt.time()
                # data: lattice0_posA lattice1_posA .... lattice0_posB lattice1_posB
                cgpt.munge_inner_outer(data_munged, data, nsingleCap, block_reduce)
                # data_munged: lattice0 lattice1 lattice2 ...
                dt_munge += gpt.time()
            else:
                data_munged = data0

            fgrid.barrier()
            dt_distr -= gpt.time()
            rhs = data_munged[0:block_data_size_single]
            distribute_plan = gpt.copy_plan(basis[0], rhs)
            distribute_plan.destination += basis[0].view[pos[b]]
            distribute_plan.source += gpt.global_memory_view(
                fgrid, [[fgrid.processor, rhs, 0, rhs.nbytes]]
            )
            rhs = None
            distribute_plan = distribute_plan()
            for i in range(nsingleCap_max):
                distribute_plan(
                    basis[i],
                    data_munged[block_data_size_single * i : block_data_size_single * (i + 1)],
                )
            dt_distr += gpt.time()

            if verbose:
                gpt.message(
                    "* read %g GB: fread at %g GB/s, crc32 at %g GB/s, munge at %g GB/s, distribute at %g GB/s; available = %g GB"
                    % (
                        totalSizeGB,
                        totalSizeGB / dt_fread,
                        totalSizeGB / dt_crc,
                        totalSizeGB / dt_munge,
                        totalSizeGB / dt_distr,
                        mem_avail(),
                    )
                )

        # fp16 data
        if nbasis != nsingleCap:
            # allocate data buffer
            data_fp32 = memoryview(bytearray(block_data_size_single * (nbasis - nsingleCap)))
            data_munged = memoryview(bytearray(block_data_size_single * (nbasis - nsingleCap)))
            for b in range(read_blocks):
                fgrid.barrier()
                dt_fread -= gpt.time()
                if f is not None:
                    data = memoryview(f.read(block_data_size_fp16 * (nbasis - nsingleCap)))
                    globalReadGB = len(data) / 1024.0**3.0
                else:
                    globalReadGB = 0.0
                globalReadGB = fgrid.globalsum(globalReadGB)
                dt_fread += gpt.time()
                totalSizeGB += globalReadGB

                if f is not None:
                    dt_crc -= gpt.time()
                    crc32_comp = gpt.crc32(data, crc32_comp)
                    dt_crc += gpt.time()
                    dt_fp16 -= gpt.time()
                    cgpt.fp16_to_fp32(data_fp32, data, 24)
                    dt_fp16 += gpt.time()
                    dt_munge -= gpt.time()
                    cgpt.munge_inner_outer(
                        data_munged,
                        data_fp32,
                        nbasis - nsingleCap,
                        block_reduce,
                    )
                    dt_munge += gpt.time()
                else:
                    data_munged = data0

                fgrid.barrier()
                dt_distr -= gpt.time()
                if nsingleCap < nbasis_max:
                    rhs = data_munged[0:block_data_size_single]
                    distribute_plan = gpt.copy_plan(basis[0], rhs)
                    distribute_plan.destination += basis[0].view[pos[b]]
                    distribute_plan.source += gpt.global_memory_view(
                        fgrid, [[fgrid.processor, rhs, 0, rhs.nbytes]]
                    )
                    rhs = None
                    distribute_plan = distribute_plan()
                    for i in range(nsingleCap, nbasis_max):
                        j = i - nsingleCap
                        distribute_plan(
                            basis[i],
                            data_munged[
                                block_data_size_single * j : block_data_size_single * (j + 1)
                            ],
                        )
                dt_distr += gpt.time()

                if verbose:
                    gpt.message(
                        "* read %g GB: fread at %g GB/s, crc32 at %g GB/s, munge at %g GB/s, distribute at %g GB/s, fp16 at %g GB/s; available = %g GB"
                        % (
                            totalSizeGB,
                            totalSizeGB / dt_fread,
                            totalSizeGB / dt_crc,
                            totalSizeGB / dt_munge,
                            totalSizeGB / dt_distr,
                            totalSizeGB / dt_fp16,
                            mem_avail(),
                        )
                    )

        # coarse grid data
        distribute_plan = None

        neigen_blocks = neigen
        coarse_block_size = coarse_vector_size
        neigen_per_block = 1

        for divisor in [2, 3, 5]:
            while neigen_blocks > max_read_blocks and neigen_blocks % divisor == 0:
                neigen_blocks //= divisor
                coarse_block_size *= divisor
                neigen_per_block *= divisor

        data_fp32 = memoryview(bytearray(coarse_fp32_vector_size * neigen_per_block))
        if verbose:
            gpt.message("Coarse read blocks", neigen_blocks)

        for j in range(neigen_blocks):
            fgrid.barrier()
            dt_fread -= gpt.time()
            if f is not None:
                data = memoryview(f.read(coarse_block_size))
                globalReadGB = len(data) / 1024.0**3.0
            else:
                globalReadGB = 0.0
            globalReadGB = fgrid.globalsum(globalReadGB)
            dt_fread += gpt.time()
            totalSizeGB += globalReadGB

            if f is not None:
                dt_crc -= gpt.time()
                crc32_comp = gpt.crc32(data, crc32_comp)
                dt_crc += gpt.time()
                dt_fp16 -= gpt.time()
                cgpt.mixed_fp32fp16_to_fp32(
                    data_fp32,
                    data,
                    coarse_block_size_part_fp32,
                    coarse_block_size_part_fp16,
                    FP16_COEF_EXP_SHARE_FLOATS,
                )
                dt_fp16 += gpt.time()
                data = data_fp32
            else:
                data = data0

            fgrid.barrier()
            dt_distr -= gpt.time()
            for l in range(neigen_per_block * j, neigen_per_block * (j + 1)):
                if l < neigen_max:
                    lidx = l - neigen_per_block * j
                    data_l = data[
                        lidx * coarse_fp32_vector_size : (lidx + 1) * coarse_fp32_vector_size
                    ]
                    if distribute_plan is None:
                        distribute_plan = gpt.copy_plan(cevec[l], data_l)
                        distribute_plan.destination += cevec[l].view[pos_coarse]
                        distribute_plan.source += gpt.global_memory_view(
                            cgrid, [[cgrid.processor, data_l, 0, data_l.nbytes]]
                        )
                        distribute_plan = distribute_plan()
                    distribute_plan(cevec[l], data_l)
            dt_distr += gpt.time()

            if verbose:  # and j % (neigen_blocks // 10) == 0
                gpt.message(
                    "* read %g GB: fread at %g GB/s, crc32 at %g GB/s, munge at %g GB/s, distribute at %g GB/s, fp16 at %g GB/s; available = %g GB"
                    % (
                        totalSizeGB,
                        totalSizeGB / dt_fread,
                        totalSizeGB / dt_crc,
                        totalSizeGB / dt_munge,
                        totalSizeGB / dt_distr,
                        totalSizeGB / dt_fp16,
                        mem_avail(),
                    )
                )

        # crc checks
        if f is not None:
            assert crc32_comp == crc32[cv.rank]

    # timing
    t1 = gpt.time()

    # verbosity
    if verbose:
        gpt.message("* load %g GB at %g GB/s" % (totalSizeGB, totalSizeGB / (t1 - t0)))

    # eigenvalues
    evln = list(filter(lambda x: x != "", open(filename + "/eigen-values.txt").read().split("\n")))
    nev = int(evln[0])
    ev = [float(x) for x in evln[1:]]
    assert len(ev) == nev
    return (basis, cevec, ev)


def save(filename, objs, params):
    # split data to save
    assert len(objs) == 3
    basis = objs[0]
    cevec = objs[1]
    ev = objs[2]

    # verbosity
    verbose = gpt.default.is_verbose("io")
    if verbose:
        gpt.message(
            "Saving %d basis vectors, %d coarse-grid vectors, %d eigenvalues to %s"
            % (len(basis), len(cevec), len(ev), filename)
        )

    # create directory
    if gpt.rank() == 0:
        os.makedirs(filename, exist_ok=True)

    # now sync since only root has created directory
    gpt.barrier()

    # write eigenvalues
    if gpt.rank() == 0:
        f = open("%s/eigen-values.txt" % filename, "wt")
        f.write("%d\n" % len(ev))
        for v in ev:
            f.write("%.15E\n" % v)
        f.close()

    # site checkerboard
    # only odd is used in this file format but
    # would be easy to generalize here
    site_cb = gpt.odd

    # grids
    assert len(basis) > 0
    assert len(cevec) > 0
    fgrid = basis[0].grid
    cgrid = cevec[0].grid

    # mpi layout
    if params["mpi"] is not None:
        mpi = params["mpi"]
    else:
        mpi = fgrid.mpi
    assert mpi[0] == 1  # assert no mpi in 5th direction

    # params
    assert basis[0].checkerboard() == site_cb
    nd = 5
    assert len(fgrid.ldimensions) == nd
    fdimensions = fgrid.fdimensions
    ldimensions = [conformDiv(fdimensions[i], mpi[i]) for i in range(nd)]
    assert fgrid.precision == gpt.single
    s = ldimensions
    b = [conformDiv(fgrid.fdimensions[i], cgrid.fdimensions[i]) for i in range(nd)]
    nb = [conformDiv(s[i], b[i]) for i in range(nd)]
    neigen = len(cevec)
    nbasis = len(basis)
    if "nsingle" in params:
        nsingle = params["nsingle"]
        assert nsingle <= nbasis
    else:
        nsingle = nbasis
    nsingleCap = min([nsingle, nbasis])
    blocks = numpy.prod(nb)
    FP16_COEF_EXP_SHARE_FLOATS = 10

    # write metadata
    if gpt.rank() == 0:
        fmeta = open("%s/metadata.txt" % filename, "wt")
        for i in range(nd):
            fmeta.write("s[%d] = %d\n" % (i, s[(i + 1) % nd]))
        for i in range(nd):
            fmeta.write("b[%d] = %d\n" % (i, b[(i + 1) % nd]))
        for i in range(nd):
            fmeta.write("nb[%d] = %d\n" % (i, nb[(i + 1) % nd]))
        fmeta.write("neig = %d\n" % neigen)
        fmeta.write("nkeep = %d\n" % nbasis)
        fmeta.write("nkeep_single = %d\n" % nsingle)
        fmeta.write("blocks = %d\n" % blocks)
        fmeta.write("FP16_COEF_EXP_SHARE_FLOATS = %d\n" % FP16_COEF_EXP_SHARE_FLOATS)
        fmeta.flush()  # write crc32 later

    # create cartesian view on fine grid
    cv0 = gpt.cartesian_view(-1, mpi, fdimensions, fgrid.cb, site_cb)
    views = cv0.views_for_node(fgrid)
    crc32 = numpy.array([0] * cv0.ranks, dtype=numpy.uint64)
    # timing
    t0 = gpt.time()
    totalSizeGB = 0
    dt_fp16 = 1e-30
    dt_distr = 1e-30
    dt_munge = 1e-30
    dt_crc = 1e-30
    dt_fwrite = 1e-30
    t0 = gpt.time()

    # load all views
    if verbose:
        gpt.message("Saving %s with %d views per node" % (filename, len(views)))

    for i, v in enumerate(views):
        cv = gpt.cartesian_view(v if v is not None else -1, mpi, fdimensions, fgrid.cb, site_cb)
        cvc = gpt.cartesian_view(
            v if v is not None else -1, mpi, cgrid.fdimensions, gpt.full, gpt.none
        )
        pos_coarse = gpt.coordinates(cvc, "canonical")

        dn, fn = get_local_name(filename, cv)
        if fn is not None:
            os.makedirs(dn, exist_ok=True)

        # sizes
        slot_lsites = numpy.prod(cv.view_dimensions)
        assert slot_lsites % blocks == 0
        block_data_size_single = slot_lsites * 12 // 2 // blocks * 2 * 4
        block_data_size_fp16 = FP_16_SIZE(slot_lsites * 12 // 2 // blocks * 2, 24)
        coarse_block_size_part_fp32 = 2 * (4 * nsingleCap)
        coarse_block_size_part_fp16 = 2 * (
            FP_16_SIZE(nbasis - nsingleCap, FP16_COEF_EXP_SHARE_FLOATS)
        )
        coarse_vector_size = (coarse_block_size_part_fp32 + coarse_block_size_part_fp16) * blocks
        totalSize = (
            blocks
            * (block_data_size_single * nsingleCap + block_data_size_fp16 * (nbasis - nsingleCap))
            + neigen * coarse_vector_size
        )
        totalSizeGB += totalSize / 1024.0**3.0 if v is not None else 0.0

        # checksum
        crc32_comp = 0

        # file
        f = gpt.FILE(fn, "wb") if fn is not None else None

        # block positions
        pos = [
            cgpt.coordinates_from_block(cv.top, cv.bottom, b, nb, "canonicalOdd")
            for b in range(blocks)
        ]

        # group blocks
        read_blocks = blocks
        block_reduce = 1
        max_read_blocks = get_param(params, "max_read_blocks", 8)
        while read_blocks > max_read_blocks and read_blocks % 2 == 0:
            pos = [
                numpy.concatenate((pos[2 * i + 0], pos[2 * i + 1])) for i in range(read_blocks // 2)
            ]
            block_data_size_single *= 2
            block_data_size_fp16 *= 2
            read_blocks //= 2
            block_reduce *= 2

        # make read-only to enable caching
        for x in pos:
            x.setflags(write=0)

        # single-precision data
        data = memoryview(bytearray(block_data_size_single * nsingleCap))
        data_munged = memoryview(bytearray(block_data_size_single * nsingleCap))

        for b in range(read_blocks):
            fgrid.barrier()
            dt_distr -= gpt.time()
            lhs_size = basis[0].otype.nfloats * 4 * len(pos[b])
            lhs = data_munged[0:lhs_size]
            distribute_plan = gpt.copy_plan(lhs, basis[0])
            distribute_plan.destination += gpt.global_memory_view(
                fgrid, [[fgrid.processor, lhs, 0, lhs.nbytes]]
            )
            distribute_plan.source += basis[0].view[pos[b]]
            distribute_plan = distribute_plan()
            lhs = None
            for i in range(nsingleCap):
                distribute_plan(
                    data_munged[block_data_size_single * i : block_data_size_single * (i + 1)],
                    basis[i],
                )
            dt_distr += gpt.time()

            if f is not None:
                dt_munge -= gpt.time()
                cgpt.munge_inner_outer(
                    data,
                    data_munged,
                    block_reduce,
                    nsingleCap,
                )
                dt_munge += gpt.time()
                dt_crc -= gpt.time()
                crc32_comp = gpt.crc32(data, crc32_comp)
                dt_crc += gpt.time()

            fgrid.barrier()
            dt_fwrite -= gpt.time()
            if f is not None:
                f.write(data)
                globalWriteGB = len(data) / 1024.0**3.0
            else:
                globalWriteGB = 0.0
            globalWriteGB = fgrid.globalsum(globalWriteGB)
            dt_fwrite += gpt.time()
            totalSizeGB += globalWriteGB

            if verbose:
                gpt.message(
                    "* write %g GB: fwrite at %g GB/s, crc32 at %g GB/s, munge at %g GB/s, distribute at %g GB/s"
                    % (
                        totalSizeGB,
                        totalSizeGB / dt_fwrite,
                        totalSizeGB / dt_crc,
                        totalSizeGB / dt_munge,
                        totalSizeGB / dt_distr,
                    )
                )

        # fp16 data
        if nbasis != nsingleCap:
            # allocate data buffer
            data_fp32 = memoryview(bytearray(block_data_size_single * (nbasis - nsingleCap)))
            data_munged = memoryview(bytearray(block_data_size_single * (nbasis - nsingleCap)))
            data = memoryview(bytearray(block_data_size_fp16 * (nbasis - nsingleCap)))
            for b in range(read_blocks):
                fgrid.barrier()
                dt_distr -= gpt.time()
                lhs_size = basis[0].otype.nfloats * 4 * len(pos[b])
                lhs = data_munged[0:lhs_size]
                distribute_plan = gpt.copy_plan(lhs, basis[0])
                distribute_plan.destination += gpt.global_memory_view(
                    fgrid, [[fgrid.processor, lhs, 0, lhs.nbytes]]
                )
                distribute_plan.source += basis[0].view[pos[b]]
                distribute_plan = distribute_plan()
                lhs = None
                for i in range(nsingleCap, nbasis):
                    j = i - nsingleCap
                    distribute_plan(
                        data_munged[j * block_data_size_single : (j + 1) * block_data_size_single],
                        basis[i],
                    )
                dt_distr += gpt.time()

                if f is not None:
                    dt_munge -= gpt.time()
                    cgpt.munge_inner_outer(
                        data_fp32,
                        data_munged,
                        block_reduce,
                        nbasis - nsingleCap,
                    )
                    dt_munge += gpt.time()
                    dt_fp16 -= gpt.time()
                    cgpt.fp32_to_fp16(data, data_fp32, 24)
                    dt_fp16 += gpt.time()
                    dt_crc -= gpt.time()
                    crc32_comp = gpt.crc32(data, crc32_comp)
                    dt_crc += gpt.time()

                fgrid.barrier()
                dt_fwrite -= gpt.time()
                if f is not None:
                    f.write(data)
                    globalWriteGB = len(data) / 1024.0**3.0
                else:
                    globalWriteGB = 0.0
                globalWriteGB = fgrid.globalsum(globalWriteGB)
                dt_fwrite += gpt.time()
                totalSizeGB += globalWriteGB

                if verbose:
                    gpt.message(
                        "* write %g GB: fwrite at %g GB/s, crc32 at %g GB/s, munge at %g GB/s, distribute at %g GB/s, fp16 at %g GB/s"
                        % (
                            totalSizeGB,
                            totalSizeGB / dt_fwrite,
                            totalSizeGB / dt_crc,
                            totalSizeGB / dt_munge,
                            totalSizeGB / dt_distr,
                            totalSizeGB / dt_fp16,
                        )
                    )

        # coarse grid data
        data = memoryview(bytearray(coarse_vector_size))
        data_fp32 = memoryview(bytearray(cevec[0].otype.nfloats * 4 * len(pos_coarse)))
        distribute_plan = gpt.copy_plan(data_fp32, cevec[0])
        distribute_plan.destination += gpt.global_memory_view(
            cgrid, [[cgrid.processor, data_fp32, 0, data_fp32.nbytes]]
        )
        distribute_plan.source += cevec[0].view[pos_coarse]
        distribute_plan = distribute_plan()
        for j in range(neigen):
            fgrid.barrier()
            dt_distr -= gpt.time()
            distribute_plan(data_fp32, cevec[j])
            dt_distr += gpt.time()

            if f is not None:
                dt_fp16 -= gpt.time()
                cgpt.fp32_to_mixed_fp32fp16(
                    data,
                    data_fp32,
                    coarse_block_size_part_fp32,
                    coarse_block_size_part_fp16,
                    FP16_COEF_EXP_SHARE_FLOATS,
                )
                dt_fp16 += gpt.time()
                dt_crc -= gpt.time()
                crc32_comp = gpt.crc32(data, crc32_comp)
                dt_crc += gpt.time()

            fgrid.barrier()
            dt_fwrite -= gpt.time()
            if f is not None:
                f.write(data)
                globalWriteGB = len(data) / 1024.0**3.0
            else:
                globalWriteGB = 0.0
            globalWriteGB = fgrid.globalsum(globalWriteGB)
            dt_fwrite += gpt.time()
            totalSizeGB += globalWriteGB

            if verbose and j % (neigen // 10) == 0:
                gpt.message(
                    "* write %g GB: fwrite at %g GB/s, crc32 at %g GB/s, munge at %g GB/s, distribute at %g GB/s, fp16 at %g GB/s"
                    % (
                        totalSizeGB,
                        totalSizeGB / dt_fwrite,
                        totalSizeGB / dt_crc,
                        totalSizeGB / dt_munge,
                        totalSizeGB / dt_distr,
                        totalSizeGB / dt_fp16,
                    )
                )

        # save crc
        crc32[cv.rank] = crc32_comp

    # synchronize crc32
    fgrid.globalsum(crc32)

    # timing
    t1 = gpt.time()

    # write crc to metadata
    if gpt.rank() == 0:
        for i in range(len(crc32)):
            fmeta.write("crc32[%d] = %X\n" % (i, crc32[i]))
        fmeta.close()

    # verbosity
    if verbose:
        gpt.message("* save %g GB at %g GB/s" % (totalSizeGB, totalSizeGB / (t1 - t0)))
