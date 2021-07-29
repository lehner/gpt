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
#define UNOP_VOID(name,opcode)						\
  case opcode: op.name(in,out); return 0.0;

#define UNOP_VOID_DAG0(name,opcode)						\
  case opcode: op.name(in,out,0); return 0.0;

#define UNOP_VOID_DAG1(name,opcode)						\
  case opcode: op.name(in,out,1); return 0.0;

#define UNOP_REALD(name,opcode)						\
  case opcode: return op.name(in,out);

#define DIRDISPOP_VOID(name,opcode)						\
  case opcode: op.name(in,out,dir,disp); return 0.0;

template<typename T>
RealD cgpt_fermion_operator_unary(T& op, int opcode, PyObject* _in,PyObject* _out) {
  typedef typename T::FermionField::vector_object vobj;

  PVector<Lattice<vobj>> in, out;
  cgpt_basis_fill(in,_in);
  cgpt_basis_fill(out,_out);
  
  switch (opcode) {
#include "register.h"
  default:
    ERR("Unknown opcode %d",opcode);
  }
}

template<typename T>
RealD cgpt_fermion_operator_dirdisp(T& op, int opcode, PyObject* _in, PyObject* _out, int dir, int disp) {
  typedef typename T::FermionField::vector_object vobj;

  PVector<Lattice<vobj>> in, out;
  cgpt_basis_fill(in,_in);
  cgpt_basis_fill(out,_out);

  switch(opcode) {
#include "register_dirdisp.h"
    default: ERR("Unknown opcode %d", opcode);
  }
}

#undef UNOP
