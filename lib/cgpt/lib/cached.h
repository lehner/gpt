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
template<typename T>
class grid_cached {
 protected:
  T* _data;
  bool _filled;
  bool _own_data;

 public:

  static void capsule_destructor(PyObject* c) {
    const char* tag = typeid(T).name();
    T* t = (T*)PyCapsule_GetPointer(c,tag);
    delete t;
  }

  grid_cached(GridBase* grid, PyArrayObject* a) {
    
    const char* tag = typeid(T).name();

    // first check if array is mutable
    bool mut = ( (PyArray_FLAGS(a) & NPY_ARRAY_WRITEABLE) != 0 );
    if (!mut) {

      // now check if cache is set
      PyObject* base = PyArray_BASE(a);
      if (base) {
	if (PyCapsule_CheckExact(base)) {
	  _data = (T*)PyCapsule_GetPointer(base,tag);
	  GridBase* cache_grid = (GridBase*)PyCapsule_GetContext(base);
	  _own_data = false;

	  if (cache_grid != grid) { // need to update
	    delete _data; _data = new T();
	    PyCapsule_SetPointer(base,_data);
	    PyCapsule_SetContext(base,grid);
	    _filled = false;
	  } else { // use cache
	    _filled = true;
	  }
	  return;
	}
      } else {
	_data = new T();
	_filled = false;
	_own_data = false;
	PyObject* cap = PyCapsule_New(_data,tag,capsule_destructor);
	PyCapsule_SetContext(cap,grid);
	PyArray_SetBaseObject(a,cap);
	return;
      }

    }

    // cannot cache
    _filled = false;
    _own_data = true;
    _data = new T();
  }

  ~grid_cached() {
    if (_own_data)
      delete _data;
  }

  T& fill_ref() {
    _filled=true;
    return *_data;
  }

  operator const T&() const {
    ASSERT(_filled);
    return *_data;
  }

  bool filled() {
    return _filled;
  }

};
