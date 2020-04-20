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
static void cgpt_to_full_coor(GridBase* grid, int cb,
			      PyArrayObject* coordinates, 
			      std::vector<cgpt_distribute::coor>& fc) {
  
  ASSERT(PyArray_NDIM(coordinates) == 2);
  long* tdim = PyArray_DIMS(coordinates);
  long nc = tdim[0];
  ASSERT(tdim[1] == grid->Nd());
  ASSERT(PyArray_TYPE(coordinates)==NPY_INT32);
  int32_t* coor = (int32_t*)PyArray_DATA(coordinates);
  long stride = grid->Nd();

  ASSERT(sizeof(Coordinate::value) == 4);

  fc.resize(nc);

  thread_region 
    {
      Coordinate site(grid->Nd());
      thread_for_in_region (i,nc,{
	  memcpy(&site[0],&coor[stride*i],4*stride);
	  ASSERT( cb == grid->CheckerBoard(site) );
	  
	  cgpt_distribute::coor& c = fc[i];
	  int odx, idx;
	  grid->GlobalCoorToRankIndex(c.rank,odx,idx,site);
	  c.offset = cgpt_distribute::offset(odx,idx);
	});
    }
}

static void cgpt_prepare_vlattice_importexport(PyObject* vlat,
					       std::vector<cgpt_distribute::data_simd> & data,
					       std::vector<long> & shape,
					       GridBase* & grid, int & cb, int & dtype) {
  ASSERT(PyList_Check(vlat));
  long nlat=PyList_Size(vlat);

  data.resize(nlat);
  shape.resize(0);

  for (long i=0;i<nlat;i++) {
    cgpt_distribute::data_simd & d = data[i];
    cgpt_Lattice_base* l = (cgpt_Lattice_base*)PyLong_AsVoidPtr(PyList_GetItem(vlat,i));
    std::vector<long> ishape;
    PyObject* _mem = l->memory_view();
    Py_buffer* buf = PyMemoryView_GET_BUFFER(_mem);
    d.local = buf->buf;
    l->describe_data_layout(d.Nsimd,d.word,d.simd_word,ishape);
    ASSERT(ishape.size() > 0 || i==0);
    
    if (i == 0) {
      shape = ishape;
      grid = l->get_grid();
      cb = l->get_checkerboard();
      dtype = l->get_numpy_dtype();
    } else {
      shape[0] += ishape[0];
      ASSERT(shape.size() == ishape.size());
      for (long j=1;j<ishape.size();j++)
	ASSERT(ishape[j]==shape[j]);
      ASSERT(grid == l->get_grid()); // only works if all lattices live on same Grid
      ASSERT(cb == l->get_checkerboard());
      ASSERT(dtype == l->get_numpy_dtype());
    }
  }
}

// Coordinates in the list may differ from node to node,
// in general they create a full map of the lattice.
static PyArrayObject* cgpt_importexport(GridBase* grid, int cb, int dtype,
					std::vector<cgpt_distribute::data_simd>& l, 
					std::vector<long> & ishape, 
					PyArrayObject* coordinates, PyObject* data) {

  cgpt_distribute dist(grid->_processor,grid->communicator);

  // distribution plan
  grid_cached<cgpt_distribute::plan> plan(grid,coordinates);
  if (!plan.filled()) {
    
    // first get full coordinates
    std::vector<cgpt_distribute::coor> fc;
    cgpt_to_full_coor(grid,cb,coordinates,fc);

    // new plan
    dist.create_plan(fc,plan.fill_ref());
  }

  // create target data layout
  long fc_size = PyArray_DIMS(coordinates)[0]; // already checked above, no need to check again
  std::vector<long> dim(1,fc_size);
  for (auto s : ishape) {
    dim.push_back(s);
  }

  if (!data) {
    // create target
    PyArrayObject* a = (PyArrayObject*)PyArray_SimpleNew((int)dim.size(), &dim[0], dtype);
    void* s = (void*)PyArray_DATA(a);
  
    // fill data
    dist.copy_to(plan,l,s);
    return a;

  } else {

    if (fc_size == 0) {
      dist.copy_from(plan,0,l);
      return 0;
    }

    // check compatibility
    void* s;
    if (PyArray_Check(data)) {
      PyArrayObject* bytes = (PyArrayObject*)data;
      ASSERT(PyArray_NDIM(bytes) == dim.size());
      long* tdim = PyArray_DIMS(bytes);
      for (int i=0;i<(int)dim.size();i++)
	ASSERT(tdim[i] == dim[i]);
      ASSERT(PyArray_TYPE(bytes) == dtype);
      s = (void*)PyArray_DATA(bytes);
    } else if (PyMemoryView_Check(data)) {
      Py_buffer* buf = PyMemoryView_GET_BUFFER(data);
      ASSERT(PyBuffer_IsContiguous(buf,'C'));
      s = (void*)buf->buf;
    } else {
      ERR("Incompatible type");
    }

    dist.copy_from(plan,s,l);
    return 0;
  }
}
