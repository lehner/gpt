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

EXPORT(create_grid,{
    
    PyObject* _gdimension, * _precision, * _type;
    if (!PyArg_ParseTuple(args, "OOO", &_gdimension, &_precision,&_type)) {
      return NULL;
    }
    
    std::vector<int> gdimension;
    std::string precision, type;
    
    cgpt_convert(_gdimension,gdimension);
    cgpt_convert(_precision,precision);
    cgpt_convert(_type,type);
    
    int nd = (int)gdimension.size();
    int Nsimd;
    
    if (precision == "single") {
      Nsimd = vComplexF::Nsimd();
    } else if (precision == "double") {
      Nsimd = vComplexD::Nsimd();
    } else {
      ERR("Unknown precision");
    }
    
    GridBase* grid;
    if (nd >= 4) {
      std::vector<int> gdimension4d(4);
      for (long i=0;i<4;i++)
	gdimension4d[i]=gdimension[nd-4+i];
      GridCartesian* grid4d = SpaceTimeGrid::makeFourDimGrid(gdimension4d, GridDefaultSimd(4,Nsimd), GridDefaultMpi());
      if (nd == 4) {
	if (type == "redblack") {
	  grid = SpaceTimeGrid::makeFourDimRedBlackGrid(grid4d);
	  delete grid4d;
	} else if (type == "full") {
	  grid = grid4d;
	} else {
	  ERR("Unknown grid type");
	}
      } else if (nd == 5) {
	if (type == "redblack") {
	  grid = SpaceTimeGrid::makeFiveDimRedBlackGrid(gdimension[0],grid4d);
	  delete grid4d;
	} else if (type == "full") {
	  grid = SpaceTimeGrid::makeFiveDimGrid(gdimension[0],grid4d);
	  delete grid4d;
	} else {
	  ERR("Unknown grid type");
	}	
      } else if (nd > 5) {
	ERR("Unknown dimension");
      }

    } else {
      // TODO: give gpt full control over mpi,simd,cbmask?
      // OR: at least give user option to make certain dimensions not simd/mpi directions
      std::cerr << "Unknown dimension " << nd << std::endl;
      assert(0);
    }
    
    return PyLong_FromVoidPtr(grid);
    
  });



EXPORT(delete_grid,{
    
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    ((GridBase*)p)->Barrier(); // before a grid goes out of life, we need to synchronize
    delete ((GridBase*)p);
    return PyLong_FromLong(0);
    
  });



EXPORT(grid_barrier,{
    
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    ((GridBase*)p)->Barrier();
    return PyLong_FromLong(0);
    
  });
  


EXPORT(grid_globalsum,{
    
    void* p;
    PyObject* o;
    if (!PyArg_ParseTuple(args, "lO", &p,&o)) {
      return NULL;
    }
    
    GridBase* grid = (GridBase*)p;
    if (PyComplex_Check(o)) {
      ComplexD c;
      cgpt_convert(o,c);
      grid->GlobalSum(c);
      return PyComplex_FromDoubles(c.real(),c.imag());
    } else if (PyFloat_Check(o)) {
      RealD c;
      cgpt_convert(o,c);
      grid->GlobalSum(c);
      return PyFloat_FromDouble(c);
    } else if (PyLong_Check(o)) {
      uint64_t c;
      cgpt_convert(o,c);
      grid->GlobalSum(c);
      return PyLong_FromLong(c);
    } else if (PyArray_Check(o)) {
      PyArrayObject* ao = (PyArrayObject*)o;
      int dt = PyArray_TYPE(ao);
      void* data = PyArray_DATA(ao);
      size_t nbytes = PyArray_NBYTES(ao);
      if (dt == NPY_FLOAT32 || dt == NPY_COMPLEX64) {
	grid->GlobalSumVector((RealF*)data, nbytes / 4);
      } else if (dt == NPY_FLOAT64 || dt == NPY_COMPLEX128) {
	grid->GlobalSumVector((RealD*)data, nbytes / 8);
      } else if (dt == NPY_UINT64) {
	grid->GlobalSumVector((uint64_t*)data, nbytes / 8);
      } else {
	ERR("Unsupported numy data type (single, double, csingle, cdouble, uint64 currently allowed)");
      }
    } else {
      ERR("Unsupported object");
    }
    // need to act on floats, complex, and numpy arrays PyArrayObject
    //PyArrayObject* p;
    return PyLong_FromLong(0);
  });

EXPORT(grid_get_processor,{
    
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    GridBase* grid = (GridBase*)p;
    long rank = grid->_processor;
    long ranks = grid->_Nprocessors;
    PyObject* coor = cgpt_convert(grid->_processor_coor);
    PyObject* ldims = cgpt_convert(grid->_ldimensions);
    PyObject* gdims = cgpt_convert(grid->_gdimensions);
    return Py_BuildValue("(l,l,N,N,N)",rank,ranks,coor,gdims,ldims);
    
  });
  
