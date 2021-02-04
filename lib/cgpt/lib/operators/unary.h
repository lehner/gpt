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
  case opcode: op.name(compatible<vobj>(in)->l,compatible<vobj>(out)->l); return 0.0;

#define UNOP_VOID_DAG0(name,opcode)						\
  case opcode: op.name(compatible<vobj>(in)->l,compatible<vobj>(out)->l,0); return 0.0;

#define UNOP_VOID_DAG1(name,opcode)						\
  case opcode: op.name(compatible<vobj>(in)->l,compatible<vobj>(out)->l,1); return 0.0;

#define UNOP_REALD(name,opcode)						\
  case opcode: return op.name(compatible<vobj>(in)->l,compatible<vobj>(out)->l);

#define DIRDISPOP_VOID(name,opcode)						\
  case opcode: op.name(compatible<vobj>(in)->l,compatible<vobj>(out)->l,dir,disp); return 0.0;

#define MULTIOP_VOID(name,opcode)						\
  case opcode: op.name(_in,in_n_virtual,_out,out_n_virtual); return 0.0;

#define MULTIOP_VOID_DAG0(name,opcode)						\
  case opcode: op.name(_in,in_n_virtual,_out,out_n_virtual,0); return 0.0;

#define MULTIOP_VOID_DAG1(name,opcode)						\
  case opcode: op.name(_in,in_n_virtual,_out,out_n_virtual,1); return 0.0;

#define MULTIOP_REALD(name,opcode)						\
  case opcode: return op.name(_in,in_n_virtual,_out,out_n_virtual);

#define MULTIDIRDISPOP_VOID(name,opcode)						\
  case opcode: op.name(_in,in_n_virtual,_out,out_n_virtual,dir,disp); return 0.0;

template<typename T>
RealD cgpt_fermion_operator_unary(T& op, int opcode, cgpt_Lattice_base* in,cgpt_Lattice_base* out) {
  typedef typename T::FermionField::vector_object vobj;
  
  switch (opcode) {
#include "register.h"
  default:
    ERR("Unknown opcode %d",opcode);
  }
}

template<typename T>
RealD cgpt_fermion_operator_dirdisp(T& op, int opcode, cgpt_Lattice_base* in, cgpt_Lattice_base* out, int dir, int disp) {
  typedef typename T::FermionField::vector_object vobj;

  switch(opcode) {
#include "register_dirdisp.h"
    default: ERR("Unknown opcode %d", opcode);
  }
}

template<typename T>
RealD cgpt_multi_arg_fermion_operator_unary(T& op, int opcode,
                                            std::vector<cgpt_Lattice_base*>& in, long in_n_virtual,
                                            std::vector<cgpt_Lattice_base*>& out, long out_n_virtual) {
  typedef typename T::BasicFermionField BasicFermionField;

  PVector<BasicFermionField> _in;
  PVector<BasicFermionField> _out;

  cgpt_basis_fill(_in,in);
  cgpt_basis_fill(_out,out);

  switch (opcode) {
#include "register_multi.h"
  default:
    ERR("Unknown opcode %d",opcode);
  }
}

template<typename T>
RealD cgpt_multi_arg_fermion_operator_dirdisp(T& op, int opcode,
                                              std::vector<cgpt_Lattice_base*>& in, long in_n_virtual,
                                              std::vector<cgpt_Lattice_base*>& out, long out_n_virtual,
                                              int dir, int disp) {
  typedef typename T::BasicFermionField BasicFermionField;

  PVector<BasicFermionField> _in;
  PVector<BasicFermionField> _out;

  cgpt_basis_fill(_in,in);
  cgpt_basis_fill(_out,out);

  switch(opcode) {
#include "register_multi_dirdisp.h"
    default: ERR("Unknown opcode %d", opcode);
  }
}

#undef UNOP
