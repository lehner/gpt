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
class cgpt_fermion_operator_base {
public:
  virtual ~cgpt_fermion_operator_base() { };
  virtual RealD unary(int opcode, cgpt_Lattice_base* in, cgpt_Lattice_base* out) = 0;
  virtual RealD dirdisp(int opcode, cgpt_Lattice_base* in, cgpt_Lattice_base* out, int dir, int disp) = 0;
  virtual void deriv(std::array<cgpt_Lattice_base*,Nd> force, cgpt_Lattice_base* X, cgpt_Lattice_base* Y, int dag) = 0;
  virtual void update(PyObject* args) = 0;
};
