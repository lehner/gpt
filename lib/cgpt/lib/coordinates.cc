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
#include "coordinates.h"

EXPORT(coordinates_from_cartesian_view,{

    PyObject* _top,* _bottom,* _checker_dim_mask,* _cb,* _order;
    if (!PyArg_ParseTuple(args, "OOOOO", &_top, &_bottom,&_checker_dim_mask,&_cb,&_order)) {
      return NULL;
    }

    std::string order;
    cgpt_convert(_order,order);
    ASSERT(PyList_Check(_top) && PyList_Check(_bottom));
    std::vector<long> top, bottom, dims, checker_dim_mask;
    std::vector<int32_t> size;
    cgpt_convert(_top,top);
    cgpt_convert(_bottom,bottom);
    cgpt_convert(_checker_dim_mask,checker_dim_mask);
    int Nd = (int)top.size();
    ASSERT(Nd == bottom.size());
    ASSERT(Nd == checker_dim_mask.size());
    long points = 1;
    for (int i=0;i<Nd;i++) {
      ASSERT(bottom[i] >= top[i]);
      size.push_back((int32_t)(bottom[i] - top[i]));
      points *= size[i];
    }

    int cb, cbf, cbd;
    long fstride = 1;
    if (Py_None == _cb) {
      cb=0;
      cbf=1;
      cbd=0;
    } else if (PyLong_Check(_cb)) {
      cb=(int)PyLong_AsLong(_cb);
      cbf=2;
      for (cbd=0;cbd<Nd;cbd++) {
	if (checker_dim_mask[cbd])
	  break;
	fstride *= size[cbd];
      }
    } else {
      ERR("Unknown argument type for _cb");
    }

    /*

      In case of even/odd lattices we may allocate too many points
      if points % cbf != 0 and first element is not on lattice:

      xox                oxo
      oxo                xox
      xox  5    versus   oxo  4

      points = 9, cbf = 2, -> allocation of (points+cbf-1)/cbf = 5 positions

    */
    dims.push_back((points+cbf-1)/cbf);
    dims.push_back(Nd);

    PyArrayObject* a = cgpt_new_PyArray((int)dims.size(), &dims[0], NPY_INT32);
    int32_t* d = (int32_t*)PyArray_DATA(a);

    bool first_on_lattice;
    if (order == "lexicographic") {

      cgpt_order_lexicographic order;
      first_on_lattice = cgpt_fill_cartesian_view_coordinates(d,Nd,top,size,checker_dim_mask,fstride,
								  cbf,cb,points,order);
      
    } else if (order == "reverse_lexicographic") {

      cgpt_order_reverse_lexicographic order;
      first_on_lattice = cgpt_fill_cartesian_view_coordinates(d,Nd,top,size,checker_dim_mask,fstride,
							      cbf,cb,points,order);

    } else if (order == "canonical") {

      Coordinate c_size = toCanonical(size);

      cgpt_order_canonical order;
      first_on_lattice = cgpt_fill_cartesian_view_coordinates(d,Nd,top,c_size,checker_dim_mask,fstride,
							      cbf,cb,points,order);

    } else {

      ERR("Unknown order scheme: %s",order.c_str());

    }

    // shrink list of coordinates if needed
    if (points % cbf != 0 && !first_on_lattice) {
      long* tdim = PyArray_DIMS(a);
      tdim[0]--;
    }

    PyArray_CLEARFLAGS(a,NPY_ARRAY_WRITEABLE); // read-only, so we can cache distribute plans
    return (PyObject*)a;
  });

EXPORT(coordinates_from_block,{

    PyObject* _top,* _bottom,* _nb,* _type;
    long block;
    if (!PyArg_ParseTuple(args, "OOlOO", &_top, &_bottom,&block,&_nb,&_type)) {
      return NULL;
    }

    ASSERT(PyList_Check(_top) && PyList_Check(_bottom) && PyList_Check(_nb));
    std::vector<long> top, bottom, nb, block_size, dims, size;
    std::string type;
    cgpt_convert(_top,top);
    cgpt_convert(_bottom,bottom);
    cgpt_convert(_nb,nb);
    cgpt_convert(_type,type);
    int Nd = (int)top.size();
    ASSERT(Nd == bottom.size());
    ASSERT(Nd == nb.size());
    long points = 1;
    for (int i=0;i<Nd;i++) {
      ASSERT(bottom[i] >= top[i]);
      size.push_back(bottom[i] - top[i]);
      ASSERT(size[i] % nb[i] == 0);
      block_size.push_back(size[i] / nb[i]);
      points *= block_size[i];
    }

    if (type == "canonicalOdd") {

      Coordinate c_nb = toCanonical(nb);
      Coordinate c_block_size = toCanonical(block_size);
      Coordinate c_top = toCanonical(top);

      dims.push_back(points/2);
      dims.push_back(Nd);

      Coordinate c_block_coor(Nd), c_block_top(Nd);
      Lexicographic::CoorFromIndex(c_block_coor,block,c_nb);
      for (int i=0;i<Nd;i++)
	c_block_top[i] = c_top[i] + c_block_coor[i] * c_block_size[i];
      
      PyArrayObject* a = cgpt_new_PyArray((int)dims.size(), &dims[0], NPY_INT32);
      int32_t* d = (int32_t*)PyArray_DATA(a);
      
      if (Nd == 5) {
	thread_region 
	  {
	    Coordinate c_coor(Nd), c_g_coor(Nd), g_coor(Nd);
	    thread_for_in_region(idx,points,{
		Lexicographic::CoorFromIndex(c_coor,idx,c_block_size);
		for (int i=0;i<Nd;i++)
		  c_g_coor[i] = c_block_top[i] + c_coor[i];
		long site_sum = c_g_coor[0] + c_g_coor[1] + c_g_coor[2] + c_g_coor[3];
		long idx_cb = idx / 2;
		if (site_sum % 2 == 1) {
		  g_coor = fromCanonical(c_g_coor);
		  for (int i=0;i<Nd;i++)
		    d[Nd*idx_cb + i] = g_coor[i];
		}
	      });
	  }
      } else {
	ERR("Dimension %d not supported",Nd);
      }
      
      PyArray_CLEARFLAGS(a,NPY_ARRAY_WRITEABLE); // read-only, so we can cache distribute plans
      return (PyObject*)a;
      
    } else {
      ERR("Unknown type %s",type.c_str());
    }
  });

EXPORT(coordinates_inserted_dimension,{

    PyObject* _coordinates, * _xdim;
    long idim;
    if (!PyArg_ParseTuple(args, "OlO", &_coordinates,&idim, &_xdim)) {
      return NULL;
    }

    std::vector<long> xdim;
    cgpt_convert(_xdim,xdim);
    ASSERT(PyArray_Check(_coordinates));
    PyArrayObject* coordinates = (PyArrayObject*)_coordinates;
    ASSERT(PyArray_TYPE(coordinates)==NPY_INT32);
    ASSERT(PyArray_NDIM(coordinates) == 2);
    long* tdim = PyArray_DIMS(coordinates);
    long nc    = tdim[0];
    long nd0   = tdim[1];
    ASSERT( 0 <= idim && idim <= nd0);
    long nd    = nd0 + 1;
    std::vector<long> dims(2);
    long xds = xdim.size();
    dims[0] = nc * xds;
    dims[1] = nd;
    PyArrayObject* a = cgpt_new_PyArray((int)dims.size(), &dims[0], NPY_INT32);
    int32_t* d = (int32_t*)PyArray_DATA(a);
    int32_t* s = (int32_t*)PyArray_DATA(coordinates);

    thread_for(ii,nc*xds,{
	long i = ii % nc;
	long l = ii / nc;
	for (long j=0;j<idim;j++)
	  d[ii*nd+j]=s[i*nd0+j];
	d[ii*nd+idim]=xdim[l];
	for (long j=idim;j<nd0;j++)
	  d[ii*nd+j+1]=s[i*nd0+j];
      });

    PyArray_CLEARFLAGS(a,NPY_ARRAY_WRITEABLE); // read-only, so we can cache distribute plans
    return (PyObject*)a;
  });

EXPORT(coordinates_momentum_phase,{

    // exp(i x mom)
    PyObject* _coordinates, * _mom, * _prec;
    if (!PyArg_ParseTuple(args, "OOO", &_coordinates,&_mom,&_prec)) {
      return NULL;
    }

    std::vector<ComplexD> mom;
    std::string prec;
    cgpt_convert(_mom,mom);
    cgpt_convert(_prec,prec);
    int dtype = infer_numpy_type(prec);
    
    ASSERT(cgpt_PyArray_Check(_coordinates));
    PyArrayObject* coordinates = (PyArrayObject*)_coordinates;
    ASSERT(PyArray_TYPE(coordinates)==NPY_INT32);
    ASSERT(PyArray_NDIM(coordinates) == 2);
    long* tdim = PyArray_DIMS(coordinates);
    long nc    = tdim[0];
    long nd    = tdim[1];
    int32_t* s = (int32_t*)PyArray_DATA(coordinates);
    ASSERT(nd == mom.size());

    std::vector<long> dims(2);
    dims[0]=nc;
    dims[1]=1;
    PyArrayObject* a = cgpt_new_PyArray((int)dims.size(),&dims[0],dtype);
    if (dtype == NPY_COMPLEX64) {
      ComplexF* d = (ComplexF*)PyArray_DATA(a);

      thread_for(i,nc,{
	  long j;
	  ComplexF arg = 0.0;
	  for (j=0;j<nd;j++) {
	    RealF x = s[i*nd+j];
	    arg+=x * (ComplexF)mom[j];
	  }
	  d[i] = exp( ComplexF(0.0,1.0)*arg );
	});

    } else if (dtype == NPY_COMPLEX128) {
      ComplexD* d = (ComplexD*)PyArray_DATA(a);

      thread_for(i,nc,{
	  long j;
	  ComplexD arg = 0.0;
	  for (j=0;j<nd;j++) {
	    RealD x = s[i*nd+j];
	    arg+=x * (ComplexD)mom[j];
	  }
	  d[i] = exp( ComplexD(0.0,1.0)*arg );
	});
    }

    return (PyObject*)a;
  });

