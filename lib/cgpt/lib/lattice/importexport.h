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
static void get_vlat_data_layout(long& sz_scalar,
				 long& sz_vector,
				 long& sz_vobj,
				 PyObject* vlat) {

  ASSERT(PyList_Check(vlat));
  long nlat=PyList_Size(vlat);
  ASSERT(nlat > 0);

  for (long i=0;i<nlat;i++) {

    cgpt_Lattice_base* l = (cgpt_Lattice_base*)PyLong_AsVoidPtr(PyList_GetItem(vlat,i));

    std::vector<long> ishape;
    long Nsimd, word, simd_word;
    l->describe_data_layout(Nsimd,word,simd_word,ishape);

    long _sz_scalar = simd_word;
    long _sz_vector = simd_word * Nsimd;
    long _sz_vobj = word * Nsimd;

    if (!i) {
      sz_scalar = _sz_scalar;
      sz_vector = _sz_vector;
      sz_vobj = _sz_vobj;
    } else {
      ASSERT(sz_scalar == _sz_scalar);
      ASSERT(sz_vector == _sz_vector);
      ASSERT(sz_vobj == _sz_vobj);
    }
  }
  
}

static void coordinates_to_memory_offsets(std::vector<long>& c_rank,
					  std::vector<long>& c_odx,
					  std::vector<long>& c_idx,
					  PyObject* vlat,
					  PyArrayObject* coordinates) {

  ASSERT(PyList_Check(vlat));
  long nlat=PyList_Size(vlat);
  ASSERT(nlat > 0);

  GridBase* grid;
  int cb;
  for (long i=0;i<nlat;i++) {

    cgpt_Lattice_base* l = (cgpt_Lattice_base*)PyLong_AsVoidPtr(PyList_GetItem(vlat,i));
    if (!i) {
      grid = l->get_grid();
      cb = l->get_checkerboard();
    } else {
      ASSERT(grid == l->get_grid());
    }
  }

  ASSERT(PyArray_NDIM(coordinates) == 2);
  long* tdim = PyArray_DIMS(coordinates);
  long nc = tdim[0];
  ASSERT(tdim[1] == grid->Nd());
  ASSERT(PyArray_TYPE(coordinates)==NPY_INT32);
  int32_t* coor = (int32_t*)PyArray_DATA(coordinates);
  long stride = grid->Nd();

  ASSERT(sizeof(Coordinate::value) == 4);

  c_rank.resize(nc);
  c_odx.resize(nc);
  c_idx.resize(nc);

  thread_region 
    {
      Coordinate site(grid->Nd());
      thread_for_in_region (i,nc,{
	  memcpy(&site[0],&coor[stride*i],4*stride);
	  ASSERT( cb == grid->CheckerBoard(site) );
	  
	  int odx, idx, rank;
	  grid->GlobalCoorToRankIndex(rank,odx,idx,site);
	  c_rank[i] = rank;
	  c_odx[i] = odx;
	  c_idx[i] = idx;
	});
    }
}


static void tensor_indices_to_memory_offsets(std::vector<long>& t_indices,
					     std::vector<long>& t_offsets,
					     PyObject* vlat,
					     long index_start, long index_stride,
					     PyArrayObject* tidx) {


  ASSERT(PyList_Check(vlat));
  long nlat=PyList_Size(vlat);

  std::vector<long> index(nlat);
  for (long i=0;i<nlat;i++)
    index[i] = index_start + i*index_stride;

  ASSERT(PyArray_TYPE(tidx) == NPY_INT32);

  ASSERT(PyArray_NDIM(tidx) == 2);
  long* tidx_dim = PyArray_DIMS(tidx);
  int32_t* tidx_coor = (int32_t*)PyArray_DATA(tidx);
  long n_indices = tidx_dim[0];
  long dim_indices = tidx_dim[1];

  t_indices.resize(n_indices);
  t_offsets.resize(n_indices);

  std::vector<long> ishape, lshape;
  long singlet_dim;
  for (long i=0;i<nlat;i++) {

    cgpt_Lattice_base* l = (cgpt_Lattice_base*)PyLong_AsVoidPtr(PyList_GetItem(vlat,i));

    long Nsimd, word, simd_word;
    std::vector<long> _lshape;
    l->describe_data_layout(Nsimd,word,simd_word,_lshape);
    ASSERT(_lshape.size() == dim_indices);

    // virtual memory layout
    int singlet_rank = l->singlet_rank();

    // allow for singlet_rank == 0 to be grouped in 1d
    if (!singlet_rank)
      singlet_rank = 1;

    if (!i) {

      singlet_dim = size_to_singlet_dim(singlet_rank, nlat);
      lshape = _lshape;
      ishape = std::vector<long>(dim_indices, singlet_dim);

    } else {

      ASSERT( singlet_dim == size_to_singlet_dim(singlet_rank, nlat) );
      for (long s=0;s<dim_indices;s++)
	ASSERT(lshape[s] == _lshape[s]);
    }
  }

  t_indices.resize(n_indices);
  t_offsets.resize(n_indices);

  thread_region
    {
      std::vector<long> gcoor(dim_indices), lcoor(dim_indices), icoor(dim_indices);
      thread_for_in_region(i,n_indices,{
	for (long l=0;l<dim_indices;l++) {
	  gcoor[l] = tidx_coor[i*dim_indices + l];
	  icoor[l] = gcoor[l] / lshape[l]; // coordinate of virtual lattice
	  lcoor[l] = gcoor[l] - icoor[l] * lshape[l]; // coordinate within virtual lattice
	}
	
	int iidx, lidx;
	Lexicographic::IndexFromCoorReversed(icoor,iidx,ishape);
	Lexicographic::IndexFromCoorReversed(lcoor,lidx,lshape);
	
	t_indices[i] = iidx;
	t_offsets[i] = lidx;
	//std::cout << i << gcoor << "= idx" <<iidx << " lidx"<<lidx<< std::endl;    
	});
    }

}

static void append_view_from_vlattice(gm_view& out,
				      PyObject* vlat,
				      long index_start, long index_stride,
				      PyArrayObject* pos,
				      PyArrayObject* tidx) {

  ASSERT(PyArray_TYPE(pos) == NPY_INT32);
  std::vector<long> t_indices, t_offsets, c_rank, c_odx, c_idx;

  // tensor dof
  tensor_indices_to_memory_offsets(t_indices, t_offsets, vlat, index_start, index_stride, tidx);

  // coordinate dof
  coordinates_to_memory_offsets(c_rank,c_odx,c_idx, vlat, pos);

  // get data layout
  long sz_scalar, sz_vector, sz_vobj;
  get_vlat_data_layout(sz_scalar,sz_vector,sz_vobj, vlat);

  // offset = idx * sz_scalar + tidx * sz_vector + odx * sz_vobj
  size_t b0 = out.blocks.size();
  out.blocks.resize(b0 + t_indices.size() * c_rank.size());

  thread_for(ci, c_rank.size(), {
      for (long i=0;i<t_indices.size();i++) {
	auto & b = out.blocks[b0 + ci * t_indices.size() + i];
	b.rank = c_rank[ci];
	b.index = index_start + t_indices[i] * index_stride;
	b.start = c_idx[ci] * sz_scalar + t_offsets[i] * sz_vector + c_odx[ci] * sz_vobj;
	b.size = sz_scalar;
      }
    });

  std::cout << t_indices << t_offsets << std::endl;
  std::cout << c_rank << std::endl;
  std::cout << c_odx << c_idx << std::endl;
  std::cout << sz_scalar << "," << sz_vector << "," << sz_vobj << std::endl;

  out.print();

  Grid_finalize();
  exit(0);

  /*    thread_region
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
  */
  
}

/*
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
    if (cgpt_PyArray_Check(data)) {
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


// direct copy
static void cgpt_importexport(GridBase* grid_dst, GridBase* grid_src, 
			      int cb_dst, int cb_src, 
			      std::vector<cgpt_distribute::data_simd>& l_dst, 
			      std::vector<cgpt_distribute::data_simd>& l_src, 
			      PyArrayObject* coordinates_dst, 
			      PyArrayObject* coordinates_src) {
  
  cgpt_distribute dist(grid_dst->_processor,grid_dst->communicator);

  // distribution plans
  grid_cached<cgpt_distribute::plan> plan_dst(grid_dst,coordinates_dst);
  grid_cached<cgpt_distribute::plan> plan_src(grid_src,coordinates_src);

  if (!plan_dst.filled()) {
    std::vector<cgpt_distribute::coor> fc;
    cgpt_to_full_coor(grid_dst,cb_dst,coordinates_dst,fc);
    dist.create_plan(fc,plan_dst.fill_ref());
  }

  if (!plan_src.filled()) {
    std::vector<cgpt_distribute::coor> fc;
    cgpt_to_full_coor(grid_src,cb_src,coordinates_src,fc);
    dist.create_plan(fc,plan_src.fill_ref());
  }

  dist.copy(plan_dst,plan_src,l_dst,l_src);
}
*/
