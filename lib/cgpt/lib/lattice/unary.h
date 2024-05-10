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
void cgpt_unary_from(Lattice<T>& dst, const Lattice<T>& src, PyObject* params) {
  ASSERT(PyDict_Check(params));
  auto op = get_str(params,"operator");
  if (op == "imag") {
    dst = imag(src);
  } else if (op == "real") {
    dst = real(src);
  } else if (op == "abs") {
    dst = cgpt_abs(src);
  } else if (op == "sqrt") {
    dst = cgpt_sqrt(src);
  } else if (op == "pow") {
    dst = cgpt_pow(src, get_float(params,"exponent"));
  } else if (op == "relu") {
    dst = cgpt_relu(src, get_float(params,"a"));
  } else if (op == "drelu") {
    dst = cgpt_drelu(src, get_float(params,"a"));
  } else if (op == "exp") {
    dst = exp(src);
  } else if (op == "log") {
    dst = cgpt_log(src);
  } else if (op == "sin") {
    dst = cgpt_sin(src);
  } else if (op == "cos") {
    dst = cgpt_cos(src);
  } else if (op == "mod") {
    dst = cgpt_mod(src, get_float(params,"n"));
  } else if (op == "tan") {
    dst = cgpt_tan(src);
  } else if (op == "asin") {
    dst = cgpt_asin(src);
  } else if (op == "acos") {
    dst = cgpt_acos(src);
  } else if (op == "atan") {
    dst = cgpt_atan(src);
  } else if (op == "sinh") {
    dst = cgpt_sinh(src);
  } else if (op == "cosh") {
    dst = cgpt_cosh(src);
  } else if (op == "tanh") {
    dst = cgpt_tanh(src);
  } else if (op == "asinh") {
    dst = cgpt_asinh(src);
  } else if (op == "acosh") {
    dst = cgpt_acosh(src);
  } else if (op == "atanh") {
    dst = cgpt_atanh(src);
  } else {
    ERR("Unknown operator %s", op.c_str());
  }
}
