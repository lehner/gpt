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
import cgpt, gpt, numpy

verbose_performance = gpt.default.is_verbose("copy_plan_performance")


class _view:
    def __init__(self, obj):
        self.obj = obj

    def __del__(self):
        cgpt.copy_delete_view(self.obj)

    def __len__(self):
        return cgpt.copy_view_size(self.obj)

    def __iadd__(self, other):
        obj_prev = self.obj
        self.obj = cgpt.copy_add_views(obj_prev, other.obj)
        cgpt.copy_delete_view(obj_prev)
        return self

    def embed_in_communicator(self, communicator):
        obj_new = cgpt.copy_view_embeded_in_communicator(self.obj, communicator.obj)
        if obj_new != 0:
            cgpt.copy_delete_view(self.obj)
            self.obj = obj_new


class copy_plan_view:
    def __init__(self, data, embed_in_communicator):
        self.view = _view(cgpt.copy_create_view(0, numpy.ndarray(shape=(0, 4), dtype=numpy.int64)))
        self.embed_in_communicator = embed_in_communicator

        n = 0
        self.memory_layout = {}
        self.data = (
            data  # keep reference to data so that id(x) stays unique for lifetime of this instance
        )
        self.requires_host_memory = False
        for x in gpt.util.to_list(data):
            self.memory_layout[id(x)] = n
            if isinstance(x, gpt.lattice):
                n += len(x.v_obj)
            elif isinstance(x, memoryview):
                n += 1
                self.requires_host_memory = True
            elif isinstance(x, numpy.ndarray):
                n += 1
                self.requires_host_memory = True
            else:
                raise Exception(f"Unknown data type {type(x)}")

    def get_index(self, x):
        x = id(x)
        assert x in self.memory_layout
        return self.memory_layout[x]

    def __iadd__(self, view_constructor):
        v = view_constructor.view(self)
        if self.embed_in_communicator is not None:
            v.embed_in_communicator(self.embed_in_communicator)
        self.view += v
        return self


class copy_plan_executer:
    def __init__(self, obj, lattice_view_location):
        self.obj = obj
        self.lattice_view_location = lattice_view_location

    def __del__(self):
        cgpt.copy_delete_plan(self.obj)

    def __call__(self, dst, src):
        dst = gpt.util.to_list(dst)
        src = gpt.util.to_list(src)
        if verbose_performance:
            t0 = gpt.time()
        cgpt.copy_execute_plan(self.obj, dst, src, self.lattice_view_location)
        if verbose_performance:
            t1 = gpt.time()
            info = [a for v in self.info().values() for a in v.values()]
            blocks = sum([a["blocks"] for a in info])
            size = sum([a["size"] for a in info])
            block_size = size // blocks
            GB = 2 * size / 1e9  # read + write = factor of 2
            gpt.message(
                f"copy_plan: execute: {GB:g} GB at {GB/(t1-t0):g} GB/s/rank with block_size {block_size}"
            )

    def info(self, details=False):
        return cgpt.copy_get_plan_info(self.obj, 1 if details else 0)


class copy_plan:
    def __init__(
        self,
        dst,
        src,
        embed_in_communicator=None,
    ):
        self.destination = copy_plan_view(dst, embed_in_communicator)
        self.source = copy_plan_view(src, embed_in_communicator)

        data_location = (
            gpt.host
            if (self.destination.requires_host_memory or self.source.requires_host_memory)
            else gpt.accelerator
        )

        self.communication_buffer_location = data_location
        self.lattice_view_location = data_location

    def __call__(self, local_only=False, skip_optimize=False, use_communication_buffers=True):
        if verbose_performance:
            cgpt.timer_begin()
        t0 = gpt.time()
        p = cgpt.copy_create_plan(
            self.destination.view.obj,
            self.source.view.obj,
            self.communication_buffer_location if use_communication_buffers else "none",
            local_only,
            skip_optimize,
        )
        t1 = gpt.time()
        if verbose_performance:
            t_cgpt = gpt.timer("cgpt_copy_create_plan", True)
            t_cgpt += cgpt.timer_end()
            gpt.message(t_cgpt)

            gpt.message(
                f"copy_plan: create: {t1-t0} s (local_only = {local_only}, skip_optimize = {skip_optimize}, use_communication_buffers = {use_communication_buffers}, communication_buffer_location = {self.communication_buffer_location.__name__})"
            )

        return copy_plan_executer(
            p,
            self.lattice_view_location,
        )


class lattice_view:
    def __init__(self, l, pos, tidx):
        self.l = gpt.util.to_list(l)
        self.pos = pos
        self.tidx = tidx

    def view(self, layout):
        v_obj = [y for x in self.l for y in x.v_obj]
        obj = cgpt.copy_create_view_from_lattice(v_obj, self.pos, self.tidx)

        # assume l are consecutive in layout, this is not checked at the moment!
        cgpt.copy_view_add_index_offset(obj, layout.get_index(self.l[0]))

        return _view(obj)


class global_memory_view:
    def __init__(self, communicator, blocks):
        self.communicator = communicator
        self.blocks = blocks

    def view(self, layout):
        if self.communicator is None:
            grid_obj = 0
        else:
            assert isinstance(self.communicator, gpt.grid)
            grid_obj = self.communicator.obj

        if self.blocks is None:
            return _view(cgpt.copy_create_view(grid_obj, None))

        processed_blocks = []
        for b in self.blocks:
            processed_blocks.append([b[0], layout.get_index(b[1]), b[2], b[3]])

        return _view(
            cgpt.copy_create_view(grid_obj, numpy.array(processed_blocks, dtype=numpy.int64))
        )
