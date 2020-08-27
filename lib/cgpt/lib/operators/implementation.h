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
class cgpt_fermion_operator : public cgpt_fermion_operator_base {
public:
  T* op;
  typedef typename T::GaugeField GaugeField;

  cgpt_fermion_operator(T* _op) : op(_op) {
  }

  virtual ~cgpt_fermion_operator() { 
    delete op;
  }

  virtual RealD unary(int opcode, cgpt_Lattice_base* in, cgpt_Lattice_base* out) {
    return cgpt_fermion_operator_unary<T>(*op,opcode,in,out);
  }

  virtual RealD dirdisp(int opcode, cgpt_Lattice_base* in, cgpt_Lattice_base* out, int dir, int disp) {
    return cgpt_fermion_operator_dirdisp<T>(*op, opcode, in, out, dir, disp);
  }

  virtual void update(PyObject* args) {
    GaugeField U(op->GaugeGrid());
    typedef typename GaugeField::vector_type vCoeff_t;
    for (int mu=0;mu<Nd;mu++) {
      auto l = get_pointer<cgpt_Lattice_base>(args,"U",mu);
      auto& Umu = compatible<iColourMatrix<vCoeff_t>>(l)->l;
      PokeIndex<LorentzIndex>(U,Umu,mu);
    }
    op->ImportGauge(U);
  }

};

template<typename T>
class cgpt_coarse_operator : public cgpt_fermion_operator_base {
public:
  T* op;
  typedef typename T::CoarseMatrix CoarseLinkField;

  cgpt_coarse_operator(T* _op) : op(_op) {
  }

  virtual ~cgpt_coarse_operator() {
    delete op;
  }

  virtual RealD unary(int opcode, cgpt_Lattice_base* in, cgpt_Lattice_base* out) {
    return cgpt_fermion_operator_unary<T>(*op,opcode,in,out);
  }

  virtual RealD dirdisp(int opcode, cgpt_Lattice_base* in, cgpt_Lattice_base* out, int dir, int disp) {
    return cgpt_fermion_operator_dirdisp<T>(*op, opcode, in, out, dir, disp);
  }

  virtual void update(PyObject* args) {
    typedef typename CoarseLinkField::vector_type vCoeff_t;
    const int nbasis = GridTypeMapper<typename T::CoarseVector::vector_object>::count;

    for(int p = 0; p < 9; p++) {
      auto l = get_pointer<cgpt_Lattice_base>(args, "A", p);
      op->A[p] = compatible<iMatrix<iSinglet<vCoeff_t>,nbasis>>(l)->l;
    }
  }
};
