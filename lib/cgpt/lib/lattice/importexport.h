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

static int get_vlat_data_layout(GridBase* & grid,
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
    l->describe_data_layout(Nsimd,word,simd_word,ishape);

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

static void append_memory_view_from_vlat(std::vector<gm_transfer::memory_view>& mv,
					 PyObject* vlat, memory_type mt, std::vector<PyObject*>& views) {

  ASSERT(PyList_Check(vlat));
  long nlat=PyList_Size(vlat);

  for (long i=0;i<nlat;i++) {
    cgpt_Lattice_base* l = (cgpt_Lattice_base*)PyLong_AsVoidPtr(PyList_GetItem(vlat,i));
    PyObject* v = l->memory_view(mt);
    views.push_back(v);

    Py_buffer* buf = PyMemoryView_GET_BUFFER(v);
    unsigned char* data = (unsigned char*)buf->buf;

    mv.push_back({ mt, data, (size_t)buf->len} );
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
	Lexicographic::IndexFromCoor(icoor,iidx,ishape); // R
	Lexicographic::IndexFromCoorReversed(lcoor,lidx,lshape); // R
	
	t_indices[i] = iidx;
	t_offsets[i] = lidx;
	//std::cout << i << gcoor << "= idx" <<iidx << " lidx"<<lidx<< std::endl;    
	});
    }

}

static PyArrayObject* create_array_to_hold_view(gm_view& dst,gm_view& src,
						PyObject* vlat,std::vector<long>& shape,
						int rank) {

  //std::cout << "ca" << std::endl;
  //std::cout << src.size() << std::endl;

  long sz_scalar, sz_vector, sz_vobj;
  GridBase* grid;
  int dtype = get_vlat_data_layout(grid, sz_scalar, sz_vector, sz_vobj, vlat);

  ASSERT(src.size() % sz_scalar == 0);
  long nelements = (long)src.size() / sz_scalar;

  long telements = 1;
  for (long i : shape)
    telements *= i;

  ASSERT(nelements % telements == 0);
  long n = nelements / telements;

  std::vector<long> fshape(1, n);
  for (long i : shape)
    fshape.push_back(i);

  dst.blocks.push_back({ rank, 0, 0, src.size() });

  //std::cout << "Elements" << fshape << std::endl;

  return (PyArrayObject*)PyArray_SimpleNew((int)fshape.size(), &fshape[0], dtype);
}

static void append_memory_view_from_dense_array(std::vector<gm_transfer::memory_view>& mv,
						PyArrayObject* d) {

  mv.push_back( { mt_host, PyArray_DATA(d), (size_t)PyArray_NBYTES(d)} );

}

static void append_memory_view_from_memory_view(std::vector<gm_transfer::memory_view>& mv,
						PyObject* d) {

  Py_buffer* buf = PyMemoryView_GET_BUFFER(d);
  mv.push_back( { mt_host, buf->buf, (size_t)buf->len} );

}

static void append_view_from_memory_view(gm_view& out,
					 PyObject* d,
					 int rank) {

  Py_buffer* buf = PyMemoryView_GET_BUFFER(d);
  out.blocks.push_back({ rank, 0, 0, (size_t)buf->len });

}

static PyObject* append_view_from_dense_array(gm_view& out,
					      PyArrayObject* data,
					      size_t sz_target,
					      int rank) {

  long ndim = PyArray_NDIM(data);
  long dtype = PyArray_TYPE(data);
  size_t sz = (size_t)PyArray_NBYTES(data);
  size_t sz_element = numpy_dtype_size(dtype);
  size_t nelements = sz / sz_element;

  out.blocks.push_back({ rank, 0, 0, sz_target });
  //std::cout << ndim << "," << dtype << "," << sz << "<>" << sz_target << std::endl;

  if (sz == sz_target) {
    Py_XINCREF(data);
  } else {
    // create new array and return it
    //std::cout << "need new array" << std::endl;

    std::vector<long> dim(1);
    ASSERT(sz_target % sz_element == 0);
    dim[0] = sz_target / sz_element;
    PyArrayObject* a = (PyArrayObject*)PyArray_SimpleNew((int)dim.size(), &dim[0], dtype);
    char* s = (char*)PyArray_DATA(data);
    char* d = (char*)PyArray_DATA(a);

    thread_for(i, dim[0], {
    	memcpy(d + sz_element*i, s + sz_element*(i % nelements), sz_element);
      });

    data = a;
  }

  return (PyObject*)data;
}

static GridBase* append_view_from_vlattice(gm_view& out,
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
  GridBase* grid;
  get_vlat_data_layout(grid, sz_scalar,sz_vector,sz_vobj, vlat);

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

  //std::cout << t_indices << t_offsets << std::endl;
  //std::cout << c_rank << std::endl;
  //std::cout << c_odx << c_idx << std::endl;
  //std::cout << sz_scalar << "," << sz_vector << "," << sz_vobj << std::endl;

  //out.print();

  return grid;
}
