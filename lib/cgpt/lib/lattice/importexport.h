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
					       std::vector<long> & shape, PyArrayObject* tidx,
					       GridBase* & grid, int & cb, int & dtype) {

  ASSERT(PyList_Check(vlat));
  long nlat=PyList_Size(vlat);

  ASSERT(PyArray_TYPE(tidx) == NPY_INT32);
  ASSERT(PyArray_NDIM(tidx) == 2);
  long* tidx_dim = PyArray_DIMS(tidx);
  int32_t* tidx_coor = (int32_t*)PyArray_DATA(tidx);
  long n_indices = tidx_dim[0];
  long dim_indices = tidx_dim[1];

  data.resize(nlat);
  shape.resize(0);

  Coordinate vcoor(dim_indices), vsize(dim_indices,1);
  Coordinate v_n0(dim_indices), v_n1(dim_indices);
  for (long i=0;i<nlat;i++) {

    cgpt_distribute::data_simd & d = data[i];

    cgpt_Lattice_base* l = (cgpt_Lattice_base*)PyLong_AsVoidPtr(PyList_GetItem(vlat,i));

    PyObject* _mem = l->memory_view();
    Py_buffer* buf = PyMemoryView_GET_BUFFER(_mem);

    d.local = buf->buf;

    std::vector<long> ishape;
    l->describe_data_layout(d.Nsimd,d.word,d.simd_word,ishape);
    ASSERT(ishape.size() == dim_indices);
    long words = d.word / d.simd_word;
    Py_XDECREF(_mem);

    // virtual memory layout
    int singlet_rank = l->singlet_rank();

    // allow for singlet_rank == 0 to be grouped in 1d
    if (!singlet_rank)
      singlet_rank = 1;

    if (!i) {


      int dim = size_to_singlet_dim(singlet_rank, nlat);
      for (long s=0;s<singlet_rank;s++)
	vsize[s]=dim;

      shape.resize(dim_indices);
      for (long s=0;s<dim_indices;s++) {
	shape[s] = ishape[s] * vsize[s];
      }

    } else {
      for (long s=singlet_rank;s<dim_indices;s++)
	ASSERT(shape[s] == ishape[s]);
    }

    Lexicographic::CoorFromIndex(vcoor,i,vsize);

    for (long s=0;s<dim_indices;s++) {
      v_n0[s] = vcoor[s] * ishape[s];
      v_n1[s] = v_n0[s] + ishape[s];
    }

    // first select
    std::vector<long> indices_on_l;
    for (long ll=0;ll<n_indices;ll++) {
      long s;
      for (s=0;s<dim_indices;s++) {
	auto & n = tidx_coor[ll*dim_indices + s];
	if (n < v_n0[s] || n >= v_n1[s])
	  break;
      }
      if (s == dim_indices)
	indices_on_l.push_back(ll);
    }

    // then process in parallel
    d.offset_data.resize(indices_on_l.size());
    d.offset_buffer.resize(indices_on_l.size());
    thread_region
      {
	std::vector<long> coor(dim_indices);
	thread_for_in_region(idx, indices_on_l.size(),{
	    for (long l=0;l<dim_indices;l++)
	      coor[l] = tidx_coor[indices_on_l[idx]*dim_indices + l] - v_n0[l];
	    int linear_index;
	    Lexicographic::IndexFromCoorReversed(coor,linear_index,ishape);
	    ASSERT(0 <= linear_index && linear_index < words);
	    d.offset_data[idx] = linear_index;
	    d.offset_buffer[idx] = indices_on_l[idx];
	  });
      }

    if (i == 0) {
      grid = l->get_grid();
      cb = l->get_checkerboard();
      dtype = l->get_numpy_dtype();
    } else {
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
					PyArrayObject* coordinates, 
					PyObject* data) {

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
      dist.copy_from(plan,0,1,l);
      return 0;
    }

    // check compatibility
    void* s;
    long sz;
    if (PyArray_Check(data)) {
      PyArrayObject* bytes = (PyArrayObject*)data;
      ASSERT(PyArray_TYPE(bytes) == dtype);
      s = (void*)PyArray_DATA(bytes);
      sz = PyArray_NBYTES(bytes);
    } else if (PyMemoryView_Check(data)) {
      Py_buffer* buf = PyMemoryView_GET_BUFFER(data);
      ASSERT(PyBuffer_IsContiguous(buf,'C'));
      s = (void*)buf->buf;
      sz = buf->len;
    } else {
      ERR("Incompatible type");
    }

    dist.copy_from(plan,s,sz,l);
    return 0;
  }
}
