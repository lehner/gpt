/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2020  Daniel Richtmann (daniel.richtmann@ur.de)

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
static PyObject* get_key(PyObject* dict, const char* key) {
  ASSERT(PyDict_Check(dict));
  PyObject* val = PyDict_GetItemString(dict,key);
  if (!val)
    ERR("Did not find parameter %s",key);
  return val;
}

template<typename T>
T* get_pointer(PyObject* dict, const char* key) {
  PyObject* val = get_key(dict,key);
  ASSERT(PyLong_Check(val));
  return (T*)PyLong_AsLong(val);
}

template<typename T>
T* get_pointer(PyObject* dict, const char* key, int mu) {
  PyObject* list = get_key(dict,key);
  ASSERT(PyList_Check(list));
  ASSERT(mu >= 0 && mu < PyList_Size(list));
  PyObject* val = PyList_GetItem(list, mu);
  ASSERT(PyLong_Check(val));
  return (T*)PyLong_AsLong(val);
}

template<typename T>
std::vector<T*> get_pointer_vec(PyObject* dict, const char* key) {
  PyObject* list = get_key(dict, key);
  ASSERT(PyList_Check(list));
  long            N = PyList_Size(list);
  std::vector<T*> ret(N);
  for(int i = 0; i < N; i++) {
    PyObject* val = PyList_GetItem(list, i);
    ASSERT(PyLong_Check(val));
    ret[i] = (T*)PyLong_AsLong(val);
  }
  return ret;
}

static RealD get_float(PyObject* dict, const char* key) {
  PyObject* _val = get_key(dict,key);
  RealD val;
  cgpt_convert(_val,val);
  return val;
}

static ComplexD get_complex(PyObject* dict, const char* key) {
  PyObject* _val = get_key(dict,key);
  ComplexD val;
  cgpt_convert(_val,val);
  return val;
}

static int get_int(PyObject* dict, const char* key) {
  PyObject* _val = get_key(dict,key);
  int val;
  cgpt_convert(_val,val);
  return val;
}

static std::string get_str(PyObject* dict, const char* key) {
  PyObject* _val = get_key(dict,key);
  std::string val;
  cgpt_convert(_val,val);
  return val;
}

static bool get_bool(PyObject* dict, const char* key) {
  PyObject* val = get_key(dict,key);
  ASSERT(PyBool_Check(val));
  return (val == Py_True);
}

static std::vector<long> get_long_vec(PyObject* dict, const char* key) {
  PyObject* val = get_key(dict,key);
  ASSERT(PyList_Check(val));
  long N = PyList_Size(val);
  std::vector<long> ret(N);
  for (int i=0;i<N;i++) {
    PyObject* _lv = PyList_GetItem(val,i);
    long lv;
    cgpt_convert(_lv,lv);
    ret[i] = lv;
  }
  return ret;
}

template<int N>
AcceleratorVector<ComplexD,N> get_complex_vec(PyObject* dict, const char* key) {
  PyObject* val = get_key(dict,key);
  ASSERT(PyList_Check(val) && PyList_Size(val) == N);

  AcceleratorVector<ComplexD,N> ret(N);
  for (int i=0;i<N;i++) {
    PyObject* _lv = PyList_GetItem(val,i);
    ComplexD lv;
    cgpt_convert(_lv,lv);
    ret[i] = lv;
  }
  return ret;
}

static std::vector<ComplexD> get_complex_vec_gen(PyObject* dict, const char* key) {
  PyObject* val = get_key(dict,key);
  ASSERT(PyList_Check(val));
  int N = (int)PyList_Size(val);

  std::vector<ComplexD> ret(N);
  for (int i=0;i<N;i++) {
    PyObject* _lv = PyList_GetItem(val,i);
    ComplexD lv;
    cgpt_convert(_lv,lv);
    ret[i] = lv;
  }
  return ret;
}
