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

  int Nsimd = grid->Nsimd();
  ASSERT(sizeof(Coordinate::value) == 4);

  fc.resize(nc);

  thread_region {
    Coordinate site(grid->Nd());
    thread_for_in_region (i,nc,{
	memcpy(&site[0],&coor[stride*i],4*stride);
	ASSERT( cb == grid->CheckerBoard(site) );
	
	cgpt_distribute::coor& c = fc[i];
	int odx, idx;
	grid->GlobalCoorToRankIndex(c.rank,odx,idx,site);
	c.offset = odx*Nsimd + idx;
      });
  }
}


// Coordinates in the list may differ from node to node,
// in general they create a full map of the lattice.
template<typename T>
PyArrayObject* cgpt_export(Lattice<T>& l, PyArrayObject* coordinates) {

  typedef typename Lattice<T>::vector_object vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename Lattice<T>::scalar_type Coeff_t;

  // infrastructure
  GridBase* grid = l.Grid();
  auto l_v = l.View();
  int Nsimd = grid->Nsimd();
  std::vector<cgpt_distribute::coor> fc;
  cgpt_distribute dist(grid->_processor,&l_v[0],sizeof(sobj),Nsimd,sizeof(Coeff_t));

  // first get full coordinates
  cgpt_to_full_coor(grid,l.Checkerboard(),coordinates,fc);

  // create target data
  std::vector<long> dim(1,fc.size());
  cgpt_numpy_data_layout(sobj(),dim);
  PyArrayObject* a = (PyArrayObject*)PyArray_SimpleNew((int)dim.size(), &dim[0], infer_numpy_type(Coeff_t()));
  sobj* s = (sobj*)PyArray_DATA(a);

  // fill data
  dist.copy_to(fc,s);

  return a;
}

template<typename T> 
void cgpt_import(Lattice<T>& l, PyArrayObject* coordinates, PyObject* data) {

  typedef typename Lattice<T>::vector_object vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename Lattice<T>::scalar_type Coeff_t;

  // infrastructure
  GridBase* grid = l.Grid();
  auto l_v = l.View();
  int Nsimd = grid->Nsimd();
  std::vector<cgpt_distribute::coor> fc;
  cgpt_distribute dist(grid->_processor,&l_v[0],sizeof(sobj),Nsimd,sizeof(Coeff_t));

  // first get full coordinates
  cgpt_to_full_coor(grid,l.Checkerboard(),coordinates,fc);

  // create target format
  std::vector<long> dim(1,fc.size());
  cgpt_numpy_data_layout(sobj(),dim);

  // check compatibility
  if (fc.size() != 0) {

    if (PyArray_Check(data)) {
      PyArrayObject* bytes = (PyArrayObject*)data;
      ASSERT(PyArray_NDIM(bytes) == dim.size());
      long* tdim = PyArray_DIMS(bytes);
      for (int i=0;i<(int)dim.size();i++)
	ASSERT(tdim[i] == dim[i]);
      ASSERT(infer_numpy_type(Coeff_t()) == PyArray_TYPE(bytes));
      sobj* s = (sobj*)PyArray_DATA(bytes);

      // fill data
      dist.copy_from(fc,s);
    } else if (PyMemoryView_Check(data)) {
      Py_buffer* buf = PyMemoryView_GET_BUFFER(data);
      ASSERT(PyBuffer_IsContiguous(buf,'C'));
      sobj* s = (sobj*)buf->buf;
      int64_t len = (int64_t)buf->len;
      ASSERT(len == sizeof(sobj)*fc.size());

      // fill data
      dist.copy_from(fc,s);
    } else {
      ERR("Incompatible type");
    }

  } else {
    dist.copy_from(fc,0);
  }

}
