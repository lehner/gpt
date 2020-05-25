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
static void cgpt_convert(PyObject* in, int& out) {
  ASSERT(PyLong_Check(in));
  out = PyLong_AsLong(in);
}

static void cgpt_convert(PyObject* in, long& out) {
  ASSERT(PyLong_Check(in));
  out = PyLong_AsLong(in);
}

static void cgpt_convert(PyObject* in, bool& out) {
  ASSERT(PyBool_Check(in));
  out = in == Py_True;
}

static bool cgpt_is_zero(PyObject* in) {
  return ((PyLong_Check(in) && PyLong_AsLong(in) == 0) ||
	  (PyFloat_Check(in) && PyFloat_AsDouble(in) == 0.0) ||
	  (PyComplex_Check(in) && PyComplex_RealAsDouble(in) == 0.0 && PyComplex_ImagAsDouble(in) == 0.0));
}

static void cgpt_convert(PyObject* in, ComplexD& out) {
  if (PyLong_Check(in)) {
    out = PyLong_AsLong(in);
  } else if (PyFloat_Check(in)) {
    out = PyFloat_AsDouble(in);
  } else if (PyComplex_Check(in)) {
    out = ComplexD(PyComplex_RealAsDouble(in),
		   PyComplex_ImagAsDouble(in));
  } else {
    ASSERT(0);
  }
}

static void cgpt_convert(PyObject* in, RealD& out) {
  if (PyLong_Check(in)) {
    out = PyLong_AsLong(in);
  } else if (PyFloat_Check(in)) {
    out = PyFloat_AsDouble(in);
  } else {
    ASSERT(0);
  }
}

static void cgpt_convert(PyObject* in, uint64_t& out) {
  if (PyLong_Check(in)) {
    out = PyLong_AsLong(in);
  } else {
    ASSERT(0);
  }
}

static void cgpt_convert(PyObject* in,  std::string& s) {
  if (PyType_Check(in)) {
    s=((PyTypeObject*)in)->tp_name;
  } else if (PyBytes_Check(in)) {
    s=PyBytes_AsString(in);
  } else if (PyUnicode_Check(in)) {
    PyObject* temp = PyUnicode_AsEncodedString(in, "UTF-8", "strict");
    ASSERT(temp);
    s=PyBytes_AS_STRING(temp);
    Py_DECREF(temp);
  } else {
    ASSERT(0);
  }
}

template<typename t>
void cgpt_convert(PyObject* in, std::vector<t>& out) {
  if (PyList_Check(in)) {
    out.resize(PyList_Size(in));
    for (size_t i = 0; i < out.size(); i++)
      cgpt_convert(PyList_GetItem(in,i),out[i]);
  } else if (PyTuple_Check(in)) {
    out.resize(PyTuple_Size(in));
    for (size_t i = 0; i < out.size(); i++)
      cgpt_convert(PyTuple_GetItem(in,i),out[i]);
  } else {
    ASSERT(0);
  }
}

static void cgpt_convert(PyObject* in, Coordinate& out) {
  if (PyList_Check(in)) {
    out.resize(PyList_Size(in));
    for (size_t i = 0; i < out.size(); i++)
      cgpt_convert(PyList_GetItem(in,i),out[i]);
  } else if (PyTuple_Check(in)) {
    out.resize(PyTuple_Size(in));
    for (size_t i = 0; i < out.size(); i++)
      cgpt_convert(PyTuple_GetItem(in,i),out[i]);
  } else {
    ASSERT(0);
  }
}

static PyObject* cgpt_convert(const Coordinate & coor) {
  PyObject* ret = PyList_New(coor.size());
  for (long i=0;i<coor.size();i++)
    PyList_SetItem(ret,i,PyLong_FromLong(coor[i]));
  return ret;
}

static Coordinate cgpt_to_coordinate(const std::vector<int>& in) {
  Coordinate out(in.size());
  for (long i=0;i<in.size();i++)
    out[i] = in[i];
  return out;
}

static std::string cgpt_str(long l) {
  char buf[64];
  sprintf(buf,"%ld",l);
  return buf;
}

static std::string cgpt_str(const Coordinate& c) {
  if (c.size() == 0)
    return "";
  std::string ret = cgpt_str(c[0]);
  for (long i=1;i<c.size();i++)
    ret += "." + cgpt_str(c[i]);
  return ret;
}
