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

static void cgpt_lattice_poke_value_internal(ComplexD& val, const std::vector<int>& in, int idx, PyObject* _val) {
  ASSERT(idx == (int)in.size());
  cgpt_convert(_val,val);
}

static void cgpt_lattice_poke_value_internal(ComplexF& val, const std::vector<int>& in, int idx, PyObject* _val) {
  ASSERT(idx == (int)in.size());
  ComplexD cv;
  cgpt_convert(_val,cv);
  val = (ComplexF)cv;
}

template<typename sobj, int N>
void cgpt_lattice_poke_value_internal(iMatrix<sobj,N>& val, const std::vector<int>& in, int idx, PyObject* _val) {
  if (idx == (int)in.size()) {
    cgpt_numpy_import(val,_val);
  } else {
    ASSERT(idx+1 < (int)in.size());
    ASSERT(in[idx] >= 0 && in[idx] < N);
    ASSERT(in[idx+1] >= 0 && in[idx+1] < N);
    cgpt_lattice_poke_value_internal(val._internal[in[idx]][in[idx+1]], in, idx+2, _val);
  }
}

template<typename sobj, int N>
void cgpt_lattice_poke_value_internal(iVector<sobj,N>& val, const std::vector<int>& in, int idx, PyObject* _val) {
  if (idx == (int)in.size()) {
    cgpt_numpy_import(val,_val);
  } else {
    ASSERT(in[idx] >= 0 && in[idx] < N);
    cgpt_lattice_poke_value_internal(val._internal[in[idx]], in, idx+1, _val);
  }
}

template<typename sobj>
void cgpt_lattice_poke_value_internal(iScalar<sobj>& val, const std::vector<int>& in, int idx, PyObject* _val) {
  if (idx == (int)in.size()) {
    cgpt_numpy_import(val,_val);
  } else {
    cgpt_lattice_poke_value_internal(val._internal, in, idx, _val);
  }
}

template<typename T>
void split_position_and_internal(Lattice<T>& l, const std::vector<int>& coor, Coordinate& pos, std::vector<int>& in) {
  GridBase* grid = l.Grid();
  pos.resize( grid->Nd() );
  ASSERT( coor.size() >= pos.size() );
  for (size_t j=0;j<pos.size();j++)
    pos[j] = coor[j];
  ASSERT( l.Checkerboard() == grid->CheckerBoard(pos) );
  for (size_t j=pos.size();j<coor.size();j++)
    in.push_back(coor[j]);
}

template<typename T>
void cgpt_lattice_poke_value(Lattice<T>& l,const std::vector<int>& coor,PyObject* _val) {

  typedef typename Lattice<T>::vector_object vobj;
  typedef typename vobj::scalar_object sobj;

  std::vector<int> in;
  Coordinate pos;
  split_position_and_internal(l, coor, pos, in);

  sobj val;
  if (!in.size()) {
    cgpt_numpy_import(val,_val);
    pokeSite(val,l,pos);
  } else {
    peekSite(val,l,pos);
    cgpt_lattice_poke_value_internal(val,in,0,_val);
    pokeSite(val,l,pos);
  }
}

template<typename sobj>
PyObject* cgpt_lattice_peek_value_internal(const iScalar<sobj>& val, const std::vector<int>& in, int idx);

static PyObject* cgpt_lattice_peek_value_internal(const ComplexD& val, const std::vector<int>& in, int idx) {
  ASSERT(idx == (int)in.size());
  return PyComplex_FromDoubles(val.real(),val.imag());
}

static PyObject* cgpt_lattice_peek_value_internal(const ComplexF& val, const std::vector<int>& in, int idx) {
  ASSERT(idx == (int)in.size());
  return PyComplex_FromDoubles(val.real(),val.imag());
}

template<typename sobj, int N>
PyObject* cgpt_lattice_peek_value_internal(const iMatrix<sobj,N>& val, const std::vector<int>& in, int idx) {
  if (idx == (int)in.size()) {
    return cgpt_numpy_export(val);
  } else {
    ASSERT(idx+1 < (int)in.size());
    ASSERT(in[idx] >= 0 && in[idx] < N);
    ASSERT(in[idx+1] >= 0 && in[idx+1] < N);
    return cgpt_lattice_peek_value_internal(val._internal[in[idx]][in[idx+1]], in, idx+2);
  }
}

template<typename sobj, int N>
PyObject* cgpt_lattice_peek_value_internal(const iVector<sobj,N>& val, const std::vector<int>& in, int idx) {
  if (idx == (int)in.size()) {
    return cgpt_numpy_export(val);
  } else {
    ASSERT(in[idx] >= 0 && in[idx] < N);
    return cgpt_lattice_peek_value_internal(val._internal[in[idx]], in, idx+1);
  }
}

template<typename sobj>
PyObject* cgpt_lattice_peek_value_internal(const iScalar<sobj>& val, const std::vector<int>& in, int idx) {
  if (idx == (int)in.size()) {
    return cgpt_numpy_export(val);
  } else {
    return cgpt_lattice_peek_value_internal(val._internal, in, idx);
  }
}

template<typename T>
PyObject* cgpt_lattice_peek_value(Lattice<T>& l,const std::vector<int>& coor) {

  typedef typename Lattice<T>::vector_object vobj;
  typedef typename vobj::scalar_object sobj;

  std::vector<int> in;
  Coordinate pos;
  split_position_and_internal(l, coor, pos, in);


  sobj val;
  peekSite(val,l,pos);
  
  if (!in.size()) {
    return cgpt_numpy_export(val);
  } else {
    return cgpt_lattice_peek_value_internal(val,in,0);
  }

}
