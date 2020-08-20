/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    
    This file tries to isolate foundational code of the data parallel layer.
    (Some of this could move to Grid.)
*/
#include <Grid/Grid.h>

using namespace Grid;

#if defined(GRID_SYCL) || defined(GRID_CUDA) || defined(GRID_HIP)
#define GRID_HAS_ACCELERATOR
#endif

NAMESPACE_BEGIN(Grid);

#if defined(GRID_CUDA)||defined(GRID_HIP)
#include "foundation/reduce_gpu.h"
#endif

#include "foundation/reduce.h"
#include "foundation/singlet.h"
#include "foundation/et.h"
#include "foundation/transfer.h"
#include "foundation/basis.h"

NAMESPACE_END(Grid);
