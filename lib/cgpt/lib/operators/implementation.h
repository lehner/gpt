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

  virtual RealD unary(int opcode, PyObject* in, PyObject* out) {
    return cgpt_fermion_operator_unary<T>(*op,opcode,in,out);
  }

  virtual RealD dirdisp(int opcode, PyObject* in, PyObject* out, int dir, int disp) {
    return cgpt_fermion_operator_dirdisp<T>(*op, opcode, in, out, dir, disp);
  }
    
  virtual RealD deriv(int opcode, PyObject* mat, PyObject* in, PyObject* out) {
    return cgpt_fermion_operator_deriv<T>(*op,opcode,mat,in,out);
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

  cgpt_coarse_operator(T* _op) : op(_op) {
  }

  virtual ~cgpt_coarse_operator() {
    delete op;
  }

  virtual RealD unary(int opcode, PyObject* in, PyObject* out) {
    return cgpt_fermion_operator_unary<T>(*op,opcode,in,out);
  }

  virtual RealD dirdisp(int opcode, PyObject* in, PyObject* out, int dir, int disp) {
    return cgpt_fermion_operator_dirdisp<T>(*op, opcode, in, out, dir, disp);
  }

  virtual RealD deriv(int opcode, PyObject* mat, PyObject* in, PyObject* out) {
    assert(0);
  }

  virtual void update(PyObject* args) {
    typedef typename T::LinkField::vector_type vCoeff_t;
    const int nbasis_virtual = GridTypeMapper<typename T::FermionField::vector_object>::count;
    
    auto ASelfInv_ptr_list = get_pointer_vec<cgpt_Lattice_base>(args,"U_self_inv");
    auto A_ptr_list = get_pointer_vec<cgpt_Lattice_base>(args,"U");
    
    PVector<Lattice<iMatrix<iSinglet<vCoeff_t>,nbasis_virtual>>> ASelfInv;
    PVector<Lattice<iMatrix<iSinglet<vCoeff_t>,nbasis_virtual>>> A;
    
    cgpt_basis_fill(ASelfInv, ASelfInv_ptr_list);
    cgpt_basis_fill(A, A_ptr_list);

    ASSERT(A.size() == 9*ASelfInv.size());

    op->ImportGauge(A, ASelfInv);
  }
};
