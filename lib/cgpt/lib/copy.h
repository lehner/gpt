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
struct cgpt_gm_view {
  Grid_MPI_Comm comm;
  int rank;
  gm_view view;
};

static cgpt_gm_view* cgpt_add_views(cgpt_gm_view* a, cgpt_gm_view* b) {
  if (a->view.blocks.size() && b->view.blocks.size()) {
    ASSERT(a->comm == b->comm);
    ASSERT(a->rank == b->rank);
  }

  cgpt_gm_view* r = new cgpt_gm_view();
  if (a->view.blocks.size()>0) {
    r->comm = a->comm;
    r->rank = a->rank;
  } else {
    r->comm = b->comm;
    r->rank = b->rank;
  }

  if (a->view.blocks.size() == 0) {
    r->view = b->view;
  } else if (b->view.blocks.size() == 0) {
    r->view = a->view;
  } else {
    
    long block_size = cgpt_gcd(a->view.block_size, b->view.block_size);

    long a_factor = a->view.block_size / block_size;
    long b_factor = b->view.block_size / block_size;
    
    r->view.blocks.resize(a_factor * a->view.blocks.size() + b_factor * b->view.blocks.size());
    r->view.block_size = block_size;

    thread_for(i, a->view.blocks.size(), {
	for (long j=0;j<a_factor;j++) {
	  auto & d = r->view.blocks[a_factor*i+j];
	  d = a->view.blocks[i];
	  d.start += j * block_size;
	}
      });
    thread_for(i, b->view.blocks.size(), {
	for (long j=0;j<b_factor;j++) {
	  auto & d = r->view.blocks[b_factor*i + j + a->view.blocks.size() * a_factor];
	  d = b->view.blocks[i];
	  d.start += j * block_size;
	}
      });
  }
  
  return r;
}

static void cgpt_view_index_offset(gm_view& v, uint32_t offset) {
  thread_for(i, v.blocks.size(), {
      auto & blk = v.blocks[i];
      blk.index += offset;
    });
}

static cgpt_gm_view* cgpt_view_embeded_in_communicator(cgpt_gm_view* v, GridBase* comm) {

  cgpt_gm_view* r = new cgpt_gm_view();
  
  if (!comm) {
    r->comm = CartesianCommunicator::communicator_world;
    r->rank = CartesianCommunicator::RankWorld();
  } else {
    r->comm = comm->communicator;
    r->rank = comm->_processor;
  }

  if (r->comm == v->comm && r->rank == v->rank)
    return 0;

  global_transfer<int> xf(v->rank, v->comm);
  std::vector<uint64_t> rank_map(xf.mpi_ranks,0);

  r->view.blocks.resize(v->view.blocks.size());
  r->view.block_size = v->view.block_size;
  
  rank_map[xf.rank] = (uint64_t)r->rank;
  xf.global_sum(rank_map);

  thread_for(i, v->view.blocks.size(), {
      auto & s = v->view.blocks[i];
      auto & d = r->view.blocks[i];
      d = s;

      // fix rank
      d.rank = (int)rank_map[s.rank];
    });

  return r;
}


static void cgpt_copy_add_memory_views(std::vector<gm_transfer::memory_view>& mv,
				       PyObject* s,
				       std::vector<PyObject*>& lattice_views,
				       memory_type lattice_view_mt) {

  ASSERT(PyList_Check(s));
  long n=PyList_Size(s);

  for (long i=0;i<n;i++) {
    PyObject* item = PyList_GetItem(s,i);
    if (cgpt_PyArray_Check(item)) {
      PyArrayObject* d = (PyArrayObject*)item;
      mv.push_back( { mt_host, PyArray_DATA(d), (size_t)PyArray_NBYTES(d)} );
    } else if (PyMemoryView_Check(item)) {
      Py_buffer* buf = PyMemoryView_GET_BUFFER(item);
      mv.push_back( { mt_host, buf->buf, (size_t)buf->len} );
    } else {
      PyObject* v_obj = PyObject_GetAttrString(item,"v_obj");
      ASSERT(v_obj && PyList_Check(v_obj));
      long nlat = PyList_Size(v_obj);

      for (long j=0;j<nlat;j++) {
	cgpt_Lattice_base* l = (cgpt_Lattice_base*)PyLong_AsVoidPtr(PyList_GetItem(v_obj,j));
	PyObject* v = l->memory_view(lattice_view_mt);
	lattice_views.push_back(v);
	
	Py_buffer* buf = PyMemoryView_GET_BUFFER(v);
	unsigned char* data = (unsigned char*)buf->buf;

	mv.push_back({ lattice_view_mt, data, (size_t)buf->len} );
      }
      Py_DECREF(v_obj);
    }
  }

}

static PyObject* cgpt_copy_cyclic_upscale_array(PyArrayObject* data,
						size_t sz_target) {

  long ndim = PyArray_NDIM(data);
  long dtype = PyArray_TYPE(data);
  size_t sz = (size_t)PyArray_NBYTES(data);
  size_t sz_element = numpy_dtype_size(dtype);
  size_t nelements = sz / sz_element;

  if (sz == sz_target) {
    Py_XINCREF(data);
  } else {
    // create new array and return it
    std::vector<long> dim(1);
    ASSERT(sz_target % sz_element == 0);
    dim[0] = sz_target / sz_element;
    PyArrayObject* a = cgpt_new_PyArray((int)dim.size(), &dim[0], dtype);
    char* s = (char*)PyArray_DATA(data);
    char* d = (char*)PyArray_DATA(a);

    thread_for(i, dim[0], {
    	memcpy(d + sz_element*i, s + sz_element*(i % nelements), sz_element);
      });

    data = a;
  }

  return (PyObject*)data;
}

static void cgpt_tensor_indices_to_memory_offsets(std::vector<long>& t_indices,
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
    l->describe_data_layout(Nsimd,word,simd_word);
    l->describe_data_shape(_lshape);
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
	Lexicographic::IndexFromCoor(icoor,iidx,ishape); // R
	Lexicographic::IndexFromCoorReversed(lcoor,lidx,lshape); // R
	
	t_indices[i] = iidx;
	t_offsets[i] = lidx;
	//std::cout << i << gcoor << "= idx" <<iidx << " lidx"<<lidx<< std::endl;    
	});
    }

}

static int cgpt_get_vlat_data_layout(GridBase* & grid,
				     long& sz_scalar,
				     long& sz_vector,
				     long& sz_vobj,
				     PyObject* vlat) {

  ASSERT(PyList_Check(vlat));
  long nlat=PyList_Size(vlat);
  int dtype;
  ASSERT(nlat > 0);

  for (long i=0;i<nlat;i++) {

    cgpt_Lattice_base* l = (cgpt_Lattice_base*)PyLong_AsVoidPtr(PyList_GetItem(vlat,i));

    std::vector<long> ishape;
    long Nsimd, word, simd_word;
    l->describe_data_layout(Nsimd,word,simd_word);
    l->describe_data_shape(ishape);

    long _sz_scalar = simd_word;
    long _sz_vector = simd_word * Nsimd;
    long _sz_vobj = word * Nsimd;

    if (!i) {
      sz_scalar = _sz_scalar;
      sz_vector = _sz_vector;
      sz_vobj = _sz_vobj;
      dtype = l->get_numpy_dtype();
      grid = l->get_grid();
    } else {
      ASSERT(sz_scalar == _sz_scalar);
      ASSERT(sz_vector == _sz_vector);
      ASSERT(sz_vobj == _sz_vobj);
      ASSERT(dtype == l->get_numpy_dtype());
      ASSERT(grid == l->get_grid());
    }
  }

  return dtype;
}

static void cgpt_coordinates_to_memory_offsets(std::vector<long>& c_rank,
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

static GridBase* cgpt_copy_append_view_from_vlattice(gm_view& out,
						     PyObject* vlat,
						     long index_start, long index_stride,
						     PyArrayObject* pos,
						     PyArrayObject* tidx) {

  ASSERT(PyArray_TYPE(pos) == NPY_INT32);
  std::vector<long> t_indices, t_offsets, c_rank, c_odx, c_idx;

  // tensor dof
  cgpt_tensor_indices_to_memory_offsets(t_indices, t_offsets, vlat, index_start, index_stride, tidx);

  // coordinate dof
  cgpt_coordinates_to_memory_offsets(c_rank,c_odx,c_idx, vlat, pos);

  // get data layout
  long sz_scalar, sz_vector, sz_vobj;
  GridBase* grid;
  cgpt_get_vlat_data_layout(grid, sz_scalar,sz_vector,sz_vobj, vlat);

  // offset = idx * sz_scalar + tidx * sz_vector + odx * sz_vobj
  size_t b0 = out.blocks.size();
  out.blocks.resize(b0 + t_indices.size() * c_rank.size());

  out.block_size = sz_scalar;
  
  thread_for(ci, c_rank.size(), {
      for (long i=0;i<t_indices.size();i++) {
	auto & b = out.blocks[b0 + ci * t_indices.size() + i];
	b.rank = c_rank[ci];
	b.index = index_start + t_indices[i] * index_stride;
	b.start = c_idx[ci] * sz_scalar + t_offsets[i] * sz_vector + c_odx[ci] * sz_vobj;
      }
    });

  //std::cout << t_indices << t_offsets << std::endl;
  //std::cout << c_rank << std::endl;
  //std::cout << c_odx << c_idx << std::endl;
  //std::cout << sz_scalar << "," << sz_vector << "," << sz_vobj << std::endl;

  //out.print();
  //std::cout << GridLogMessage << "LV " << out.block_size << " , " << out.blocks.size() << std::endl;
  return grid;
}
