/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2021  Mattia Bruno

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
#define DERIVOP_VOID_DAG0(name,opcode)           \
  case opcode: op.name(frc,u,v,0); break;

#define DERIVOP_VOID_DAG1(name,opcode)           \
  case opcode: op.name(frc,u,v,1); break;

template<typename T>
RealD cgpt_fermion_operator_deriv(T& op, int opcode, PyObject* _mat, PyObject* _u, PyObject* _v) {
  typedef typename T::FermionField::vector_object vobj;
  typedef typename T::GaugeField GaugeField;

  PVector<Lattice<vobj>> u, v;
  cgpt_basis_fill(u,_u);
  cgpt_basis_fill(v,_v);

  typedef typename GaugeField::vector_type vCoeff_t;
  PVector<Lattice<iColourMatrix<vCoeff_t>>> mat;
  cgpt_basis_fill(mat, _mat);

  GaugeField frc(mat[0].Grid());
  frc = Zero();
    
  switch(opcode) {
#include "register_deriv.h"
    default: ERR("Unknown opcode %d", opcode);
  }

  for (int mu=0;mu<Nd;mu++)
    mat[mu] = PeekIndex<LorentzIndex>(frc,mu);

  return 0.0;
}

#undef UNOP