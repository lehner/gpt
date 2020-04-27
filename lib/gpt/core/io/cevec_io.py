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

# get local dir an filename
def get_local_name(root, cv):
    if cv.rank < 0:
        return None,None
    ntotal=cv.ranks
    rank=cv.rank
    dirs=32
    nperdir = ntotal // dirs
    if nperdir < 1:
        nperdir=1
    dirrank=rank//nperdir
    directory = "%s/%2.2d" % (root,dirrank)
    filename="%s/%10.10d.compressed" % (directory,rank)
    return directory,filename

def FP_16_SIZE(a,b):
    assert(a % b == 0)
    return (( (a) + (a//b) )*2)

def read_metadata(fn):
    return dict([ tuple([ x.strip() for x in ln.split("=")]) for ln in filter(lambda x: x!="" and x[0] != "#",open(fn).read().split("\n")) ])

def get_vec(d,n,conv):
    i=0
    r=[]
    while True:
        tag="%s[%d]" % (n,i)
        i+=1
        if tag in d:
            r.append(conv(d[tag]))
        else:
            return r

def get_ivec(d,n):
    return get_vec(d,n,int)

def get_xvec(d,n):
    return get_vec(d,n,lambda x: int(x,16))

def load(filename, *a):

    # first check if this is right file format
    if (not os.path.exists(filename + "/00/0000000000.compressed") or
        not os.path.exists(filename + "/metadata.txt")):
        raise NotImplementedError()

    # parameters
    if len(a) == 0:
        params={}
    else:
        params=a[0]

    # verbosity
    verbose = gpt.default.is_verbose("io")

    # site checkerboard
    # only odd is used in this file format but
    # would be easy to generalize here
    site_cb = gpt.odd

    # need grids parameter
    assert("grids" in params)
    assert(type(params["grids"]) == gpt.grid)
    fgrid=params["grids"]
    assert(fgrid.precision == gpt.single)
    fdimensions=fgrid.fdimensions

    # read metadata
    metadata=read_metadata(filename + "/metadata.txt")
    s=get_ivec(metadata,"s")
    ldimensions=[ s[4] ] + s[:4]
    blocksize=get_ivec(metadata,"b")
    blocksize=[ blocksize[4] ] + blocksize[:4]
    nb=get_ivec(metadata,"nb")
    nb=[ nb[4] ] + nb[:4]
    crc32=get_xvec(metadata,"crc32")
    neigen=int(metadata["neig"])
    nbasis=int(metadata["nkeep"])
    nsingle=int(metadata["nkeep_single"])
    blocks=int(metadata["blocks"])
    FP16_COEF_EXP_SHARE_FLOATS=int(metadata["FP16_COEF_EXP_SHARE_FLOATS"])
    nsingleCap=min([nsingle,nbasis])
	
    # check
    nd=len(ldimensions)
    assert(nd == 5)
    assert(nd == len(fdimensions))
    assert(nd == len(blocksize))
    assert(fgrid.cb == gpt.redblack)

    # create coarse grid
    cgrid=gpt.block.grid(fgrid,blocksize)

    # allocate all lattices
    basis=[ gpt.vspincolor(fgrid) for i in range(nbasis) ]
    cevec=[ gpt.vcomplex(cgrid,nbasis) for i in range(neigen) ]

    # fix checkerboard of basis
    for i in range(nbasis):
        basis[i].checkerboard(site_cb)

    # mpi layout
    mpi=[]
    for i in range(nd):
        assert(fdimensions[i] % ldimensions[i] == 0)
        mpi.append(fdimensions[i] // ldimensions[i])
    assert(mpi[0] == 1) # assert no mpi in 5th direction

    # create cartesian view on fine grid
    cv0=gpt.cartesian_view(-1,mpi,fdimensions,fgrid.cb,site_cb)
    views=cv0.views_for_node(fgrid)

    # timing
    totalSizeGB=0
    dt_fp16=1e-30
    dt_distr=1e-30
    dt_munge=1e-30
    dt_crc=1e-30
    dt_fread=1e-30
    t0=gpt.time()

    # load all views
    if verbose:
        gpt.message("Loading %s with %d views per node" % (filename,len(views)))
    for i,v in enumerate(views):
        cv=gpt.cartesian_view(v if not v is None else -1,mpi,fdimensions,fgrid.cb,site_cb)
        cvc=gpt.cartesian_view(v if not v is None else -1,mpi,cgrid.fdimensions,gpt.full,gpt.none)
        pos_coarse=gpt.coordinates(cvc,"canonical")

        dn,fn=get_local_name(filename,cv)

        # sizes
        slot_lsites=numpy.prod(cv.view_dimensions)
        assert(slot_lsites % blocks == 0)
        block_data_size_single=slot_lsites * 12 // 2 // blocks * 2 * 4
        block_data_size_fp16=FP_16_SIZE(slot_lsites * 12 // 2 // blocks * 2, 24)
        coarse_block_size_part_fp32=2*(4*nsingleCap)
        coarse_block_size_part_fp16=2*(FP_16_SIZE(nbasis-nsingleCap, FP16_COEF_EXP_SHARE_FLOATS))
        coarse_vector_size=(coarse_block_size_part_fp32+coarse_block_size_part_fp16)*blocks
        coarse_fp32_vector_size=2*(4*nbasis)*blocks
    
        # checksum
        crc32_comp=0
        
        # file
        f=gpt.FILE(fn,"rb") if not fn is None else None

        # block positions
        pos=[ cgpt.coordinates_from_block(cv.top,cv.bottom,b,nb,"canonicalOdd") for b in range(blocks) ]

        # group blocks
        read_blocks=blocks
        block_reduce=1
        max_read_blocks=8
        while read_blocks > max_read_blocks and read_blocks % 2 == 0:
            pos=[ numpy.concatenate( (pos[2*i+0],pos[2*i+1]) ) for i in range(read_blocks // 2) ]
            block_data_size_single *= 2
            block_data_size_fp16 *= 2
            read_blocks //= 2
            block_reduce *= 2

        # make read-only to enable caching
        for x in pos:
            x.setflags(write=0)

        # dummy buffer
        data0=memoryview(bytes())

        # single-precision data
        data_munged=memoryview(bytearray(block_data_size_single*nsingleCap))
        reduced_size=len(data_munged) // block_reduce
        for b in range(read_blocks):
            fgrid.barrier()
            dt_fread-=gpt.time()
            if not f is None:
                data=memoryview(f.read(block_data_size_single * nsingleCap))
                globalReadGB=len(data) / 1024.**3.
            else:
                globalReadGB=0.0
            globalReadGB=fgrid.globalsum(globalReadGB)
            dt_fread+=gpt.time()
            totalSizeGB+=globalReadGB

            if not f is None:
                dt_crc-=gpt.time()
                crc32_comp=gpt.crc32(data,crc32_comp)
                dt_crc+=gpt.time()
                dt_munge-=gpt.time()
                for l in range(block_reduce):
                    cgpt.munge_inner_outer(data_munged[reduced_size*l:reduced_size*(l+1)],
                                           data[reduced_size*l:reduced_size*(l+1)],
                                           len(pos[b]) // block_reduce,
                                           nsingleCap)
                dt_munge+=gpt.time()
            else:
                data_munged=data0

            fgrid.barrier()
            dt_distr-=gpt.time()
            gpt.poke(basis[0:nsingleCap],pos[b],data_munged)
            dt_distr+=gpt.time()

            if verbose:
                gpt.message("* read %g GB: fread at %g GB/s, crc32 at %g GB/s, munge at %g GB/s, distribute at %g GB/s" % 
                            (totalSizeGB,totalSizeGB/dt_fread,totalSizeGB/dt_crc,totalSizeGB/dt_munge,totalSizeGB/dt_distr))


        # fp16 data
        if nbasis != nsingleCap:
            # allocate data buffer
            data_fp32 = memoryview(bytearray(block_data_size_single * (nbasis-nsingleCap)))
            data_munged= memoryview(bytearray(block_data_size_single * (nbasis-nsingleCap)))
            reduced_size=len(data_munged) // block_reduce
            for b in range(read_blocks):
                fgrid.barrier()
                dt_fread-=gpt.time()
                if not f is None:
                    data=memoryview(f.read(block_data_size_fp16 * (nbasis-nsingleCap)))
                    globalReadGB=len(data) / 1024.**3.
                else:
                    globalReadGB=0.0
                globalReadGB=fgrid.globalsum(globalReadGB)
                dt_fread+=gpt.time()
                totalSizeGB+=globalReadGB

                if not f is None:
                    crc32_comp=gpt.crc32(data,crc32_comp)
                    dt_fp16-=gpt.time()
                    cgpt.fp16_to_fp32(data_fp32,data,24)
                    dt_fp16+=gpt.time()
                    dt_munge-=gpt.time()
                    for l in range(block_reduce):
                        cgpt.munge_inner_outer(data_munged[reduced_size*l:reduced_size*(l+1)],
                                               data_fp32[reduced_size*l:reduced_size*(l+1)],
                                               len(pos[b]) // block_reduce,
                                               nsingleCap)
                    dt_munge+=gpt.time()
                else:
                    data_munged=data0

                fgrid.barrier()
                dt_distr-=gpt.time()
                gpt.poke(basis[nsingleCap:nbasis],pos[b],data_munged)
                dt_distr+=gpt.time()

                if verbose:
                    gpt.message("* read %g GB: fread at %g GB/s, crc32 at %g GB/s, munge at %g GB/s, distribute at %g GB/s, fp16 at %g GB/s" % 
                                (totalSizeGB,totalSizeGB/dt_fread,totalSizeGB/dt_crc,totalSizeGB/dt_munge,totalSizeGB/dt_distr,totalSizeGB/dt_fp16))


        # coarse grid data
        data_fp32=memoryview(bytearray(coarse_fp32_vector_size))
        for j in range(neigen):
            fgrid.barrier()
            dt_fread-=gpt.time()
            if not f is None:
                data=memoryview(f.read(coarse_vector_size))
                globalReadGB=len(data) / 1024.**3.
            else:
                globalReadGB=0.0
            globalReadGB=fgrid.globalsum(globalReadGB)
            dt_fread+=gpt.time()
            totalSizeGB+=globalReadGB

            if not f is None:
                dt_crc-=gpt.time()
                crc32_comp=gpt.crc32(data,crc32_comp)
                dt_crc+=gpt.time()
                dt_fp16-=gpt.time()
                cgpt.mixed_fp32fp16_to_fp32(data_fp32,data,coarse_block_size_part_fp32,coarse_block_size_part_fp16,
                                            FP16_COEF_EXP_SHARE_FLOATS)
                dt_fp16+=gpt.time()
                data=data_fp32
            else:
                data=data0

            fgrid.barrier()
            dt_distr-=gpt.time()
            cevec[j][pos_coarse]=data
            dt_distr+=gpt.time()

            if verbose and j % (neigen // 10) == 0:
                gpt.message("* read %g GB: fread at %g GB/s, crc32 at %g GB/s, munge at %g GB/s, distribute at %g GB/s, fp16 at %g GB/s" % 
                            (totalSizeGB,totalSizeGB/dt_fread,totalSizeGB/dt_crc,totalSizeGB/dt_munge,totalSizeGB/dt_distr,totalSizeGB/dt_fp16))

        # crc checks
        if not f is None:
            assert(crc32_comp == crc32[cv.rank])

    # timing
    t1=gpt.time()

    # verbosity
    if verbose:
        gpt.message("* load %g GB at %g GB/s" % (totalSizeGB,totalSizeGB/(t1-t0)))

    # eigenvalues
    evln=list(filter(lambda x: x!="",open(filename + "/eigen-values.txt").read().split("\n")))
    nev=int(evln[0])
    ev=[ float(x) for x in evln[1:] ]
    assert(len(ev) == nev)
    return (basis,cevec,ev)


def save(filename, objs, params):

    # split data to save
    assert(len(objs) == 3)
    basis=objs[0]
    cevec=objs[1]
    ev=objs[2]

    # verbosity
    verbose = gpt.default.is_verbose("io")
    if verbose:
        gpt.message("Saving %d basis vectors, %d coarse-grid vectors, %d eigenvalues to %s" %
                    (len(basis),len(cevec),len(ev),filename))

    # create directory
    if gpt.rank() == 0:
        os.makedirs(filename, exist_ok=True)

    # now sync since only root has created directory
    gpt.barrier()

    # write eigenvalues
    f=open("%s/eigen-values.txt" % filename,"wt")
    f.write("%d\n" % len(ev))
    for v in ev:
        f.write("%.15E\n" % v)
    f.close()

    # write metadata
    fmeta=open("%s/metadata.txt" % filename,"wt")
    #s[0] = 16
    #s[1] = 16
    #s[2] = 16
    #s[3] = 8
    #s[4] = 12
    #b[0] = 2
    #b[1] = 2
    #b[2] = 2
    #b[3] = 2
    #b[4] = 12
    #nb[0] = 8
    #nb[1] = 8
    #nb[2] = 8
    #nb[3] = 4
    #nb[4] = 1
    #neig = 60
    #nkeep = 60
    #nkeep_single = 30
    #blocks = 2048
    #FP16_COEF_EXP_SHARE_FLOATS = 10
    #crc32[0] = 89F7E201


    return


    # site checkerboard
    # only odd is used in this file format but
    # would be easy to generalize here
    site_cb = gpt.odd

    # need grids parameter
    assert("grids" in params)
    assert(type(params["grids"]) == gpt.grid)
    fgrid=params["grids"]
    assert(fgrid.precision == gpt.single)
    fdimensions=fgrid.fdimensions

    # read metadata
    metadata=read_metadata(filename + "/metadata.txt")
    s=get_ivec(metadata,"s")
    ldimensions=[ s[4] ] + s[:4]
    blocksize=get_ivec(metadata,"b")
    blocksize=[ blocksize[4] ] + blocksize[:4]
    nb=get_ivec(metadata,"nb")
    nb=[ nb[4] ] + nb[:4]
    crc32=get_xvec(metadata,"crc32")
    neigen=int(metadata["neig"])
    nbasis=int(metadata["nkeep"])
    nsingle=int(metadata["nkeep_single"])
    blocks=int(metadata["blocks"])
    FP16_COEF_EXP_SHARE_FLOATS=int(metadata["FP16_COEF_EXP_SHARE_FLOATS"])
    nsingleCap=min([nsingle,nbasis])
	
    # check
    nd=len(ldimensions)
    assert(nd == 5)
    assert(nd == len(fdimensions))
    assert(nd == len(blocksize))
    assert(fgrid.cb == gpt.redblack)

    # create coarse grid
    cgrid=gpt.block.grid(fgrid,blocksize)

    # allocate all lattices
    basis=[ gpt.vspincolor(fgrid) for i in range(nbasis) ]
    cevec=[ gpt.vcomplex(cgrid,nbasis) for i in range(neigen) ]

    # fix checkerboard of basis
    for i in range(nbasis):
        basis[i].checkerboard(site_cb)

    # mpi layout
    mpi=[]
    for i in range(nd):
        assert(fdimensions[i] % ldimensions[i] == 0)
        mpi.append(fdimensions[i] // ldimensions[i])
    assert(mpi[0] == 1) # assert no mpi in 5th direction

    # create cartesian view on fine grid
    cv0=gpt.cartesian_view(-1,mpi,fdimensions,fgrid.cb,site_cb)
    views=cv0.views_for_node(fgrid)

    # timing
    t0=gpt.time()
    totalSizeGB=0
    dt_fp16=0.0
    nb_fp16=0.0
    dt_file=0.0
    dt_distr_c=0.0
    dt_distr_f=0.0

    # load all views
    if verbose:
        gpt.message("Loading %s with %d views per node" % (filename,len(views)))
    for i,v in enumerate(views):
        cv=gpt.cartesian_view(v if not v is None else -1,mpi,fdimensions,fgrid.cb,site_cb)
        cvc=gpt.cartesian_view(v if not v is None else -1,mpi,cgrid.fdimensions,gpt.full,gpt.none)
        pos_coarse=gpt.coordinates(cvc,"canonical")

        dn,fn=get_local_name(filename,cv)

        # sizes
        slot_lsites=numpy.prod(cv.view_dimensions)
        assert(slot_lsites % blocks == 0)
        block_data_size_single=slot_lsites * 12 // 2 // blocks * 2 * 4
        block_data_size_fp16=FP_16_SIZE(slot_lsites * 12 // 2 // blocks * 2, 24)
        coarse_block_size_part_fp32=2*(4*nsingleCap)
        coarse_block_size_part_fp16=2*(FP_16_SIZE(nbasis-nsingleCap, FP16_COEF_EXP_SHARE_FLOATS))
        coarse_vector_size=(coarse_block_size_part_fp32+coarse_block_size_part_fp16)*blocks
        coarse_fp32_vector_size=2*(4*nbasis)*blocks
        totalSize=blocks*(block_data_size_single * nsingleCap + block_data_size_fp16  * (nbasis-nsingleCap)) + neigen * coarse_vector_size
        totalSizeGB+=totalSize / 1024.**3. if not v is None else 0.0

        # checksum
        crc32_comp=0
        
        # file
        f=gpt.FILE(fn,"r+b") if not fn is None else None

        # block positions
        pos=[ cgpt.coordinates_from_block(cv.top,cv.bottom,b,nb,"canonicalOdd") for b in range(blocks) ]

        # group blocks
        read_blocks=blocks
        block_reduce=1
        max_read_blocks=16
        while read_blocks > max_read_blocks and read_blocks % 2 == 0:
            pos=[ numpy.concatenate( (pos[2*i+0],pos[2*i+1]) ) for i in range(read_blocks // 2) ]
            block_data_size_single *= 2
            block_data_size_fp16 *= 2
            read_blocks //= 2
            block_reduce *= 2

        # make read-only to enable caching
        for x in pos:
            x.setflags(write=0)

        # dummy buffer
        data0=memoryview(bytes())

        # single-precision data
        data_munged=memoryview(bytearray(block_data_size_single*nsingleCap))
        reduced_size=len(data_munged) // block_reduce
        for b in range(read_blocks):
            if not f is None:
                data=memoryview(f.read(block_data_size_single * nsingleCap))
                crc32_comp=gpt.crc32(data,crc32_comp)
                for l in range(block_reduce):
                    cgpt.munge_inner_outer(data_munged[reduced_size*l:reduced_size*(l+1)],
                                           data[reduced_size*l:reduced_size*(l+1)],
                                           len(pos[b]) // block_reduce,
                                           nsingleCap)
            else:
                data_munged=data0
            dt_distr_f-=gpt.time()
            gpt.poke(basis[0:nsingleCap],pos[b],data_munged)
            dt_distr_f+=gpt.time()

        # fp16 data
        if nbasis != nsingleCap:
            # allocate data buffer
            data_fp32 = memoryview(bytearray(block_data_size_single * (nbasis-nsingleCap)))
            data_munged= memoryview(bytearray(block_data_size_single * (nbasis-nsingleCap)))
            reduced_size=len(data_munged) // block_reduce
            for b in range(read_blocks):
                if not f is None:
                    data=memoryview(f.read(block_data_size_fp16 * (nbasis-nsingleCap)))
                    crc32_comp=gpt.crc32(data,crc32_comp)
                    dt_fp16-=gpt.time()
                    cgpt.fp16_to_fp32(data_fp32,data,24)
                    dt_fp16+=gpt.time()
                    nb_fp16+=block_data_size_fp16 * (nbasis-nsingleCap) * 2

                    for l in range(block_reduce):
                        cgpt.munge_inner_outer(data_munged[reduced_size*l:reduced_size*(l+1)],
                                               data_fp32[reduced_size*l:reduced_size*(l+1)],
                                               len(pos[b]) // block_reduce,
                                               nsingleCap)

                else:
                    data_munged=data0
                dt_distr_f-=gpt.time()
                gpt.poke(basis[nsingleCap:nbasis],pos[b],data_munged)
                dt_distr_f+=gpt.time()

        # coarse grid data
        data_fp32=memoryview(bytearray(coarse_fp32_vector_size))
        for j in range(neigen):
            if not f is None:
                data=memoryview(f.read(coarse_vector_size))
                crc32_comp=gpt.crc32(data,crc32_comp)
                cgpt.mixed_fp32fp16_to_fp32(data_fp32,data,coarse_block_size_part_fp32,coarse_block_size_part_fp16,
                                            FP16_COEF_EXP_SHARE_FLOATS)
                data=data_fp32
            else:
                data=data0
            dt_distr_c-=gpt.time()
            cevec[j][pos_coarse]=data
            dt_distr_c+=gpt.time()

        # crc checks
        if not f is None:
            assert(crc32_comp == crc32[cv.rank])

    # timing
    t1=gpt.time()
    totalSizeGB=fgrid.globalsum(totalSizeGB)

    # test
    #for j in range(len(basis)):
    #    gpt.message(j,gpt.norm2(basis[j]),cv0.ranks,blocks)
            
    # verbosity
    if verbose:
        gpt.message("* load %g GB at %g GB/s" % (totalSizeGB,totalSizeGB/(t1-t0)))
        gpt.message("* total: %g s, distribute coarse: %g s, distribute fine: %g s" % (t1-t0,dt_distr_c,dt_distr_f))
        nb_fp16GB=fgrid.globalsum(nb_fp16) / 1024.**3.
        if nb_fp16GB != 0.0 and dt_fp16 != 0.0:
            gpt.message("* converted FP16 to FP32 at %g GB/s" % (nb_fp16GB/dt_fp16))

    # eigenvalues
    evln=list(filter(lambda x: x!="",open(filename + "/eigen-values.txt").read().split("\n")))
    nev=int(evln[0])
    ev=[ float(x) for x in evln[1:] ]
    assert(len(ev) == nev)
    return (basis,cevec,ev)
