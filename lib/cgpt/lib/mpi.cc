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
*/
#include "lib.h"

EXPORT(global_rank,{
    return PyLong_FromLong(CartesianCommunicator::RankWorld());
  });

EXPORT(global_ranks,{
    int mpi_ranks;
#ifdef CGPT_USE_MPI
    MPI_Comm_size(CartesianCommunicator::communicator_world,&mpi_ranks);
#else
    mpi_ranks=1;
#endif
    return PyLong_FromLong(mpi_ranks);
  });

EXPORT(broadcast,{

    long root;
    PyObject* data;
    if (!PyArg_ParseTuple(args, "lO", &root,&data)) {
      return NULL;
    }

    long rank = CartesianCommunicator::RankWorld();
    if (root == rank) {
      ASSERT(PyBytes_Check(data));
      long sz = PyBytes_Size(data);
      char* p = PyBytes_AsString(data);
      ASSERT(sz < INT_MAX);
      CartesianCommunicator::BroadcastWorld((int)root,&sz,sizeof(long));
      CartesianCommunicator::BroadcastWorld((int)root,(void*)p,(int)sz);

      Py_XINCREF(data);
      return data;

    } else {
      long sz;
      CartesianCommunicator::BroadcastWorld((int)root,&sz,sizeof(long));
      char* p = new char[sz];
      CartesianCommunicator::BroadcastWorld((int)root,(void*)p,(int)sz);

      PyObject* ret = PyBytes_FromStringAndSize(p,sz);
      delete[] p; // inefficient but do not know how to construct bytes object with my memory
      return ret;
    }

  });

EXPORT(barrier,{

    #if defined (GRID_COMMS_MPI3)
    MPI_Barrier(CartesianCommunicator::communicator_world);
    #endif

    return PyLong_FromLong(0);

  });
