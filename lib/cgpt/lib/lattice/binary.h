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
void cgpt_binary_from(Lattice<T>& dst, const Lattice<T>& a, const Lattice<T>& b, PyObject* params) {
  ASSERT(PyDict_Check(params));
  auto op = get_str(params,"operator");
  if (op == "<") {
    cgpt_lower_than(dst, a, b);
  } else if (op == "*") {
    cgpt_component_wise_multiply(dst, a, b);
  } else {
    ERR("Unknown operator %s", op.c_str());
  }
}
