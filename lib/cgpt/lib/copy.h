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

static memory_type cgpt_memory_type_from_string(const std::string& s) {
  if (s == "none") {
    return mt_none;
  } else if (s == "host") {
    return mt_host;
  } else if (s == "accelerator") {
    return mt_accelerator;
  } else {
    ERR("Unknown memory_type %s",s.c_str());
  }
}

static void cgpt_copy_add_memory_views(std::vector<gm_transfer::memory_view>& mv,
				       PyObject* s,
				       std::vector<PyObject*>& lattice_views) {

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
      ASSERT(PyList_Check(item));
      ASSERT(PyList_Size(item) == 2);
      PyObject* _tbuffer = PyList_GetItem(item,0);
      std::string tbuffer;
      cgpt_convert(_tbuffer,tbuffer);
      cgpt_Lattice_base* l = (cgpt_Lattice_base*)PyLong_AsVoidPtr(PyList_GetItem(item,1));
      memory_type mt = cgpt_memory_type_from_string(tbuffer);
      PyObject* v = l->memory_view(mt);
      lattice_views.push_back(v);

      Py_buffer* buf = PyMemoryView_GET_BUFFER(v);
      unsigned char* data = (unsigned char*)buf->buf;

      mv.push_back({ mt, data, (size_t)buf->len} );
    }
  }

}
