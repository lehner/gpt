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
      auto l = get_pointer<cgpt_Lattice_base>(args, "U", p);
      op->A[p] = compatible<iMatrix<iSinglet<vCoeff_t>,nbasis>>(l)->l;
    }
  }
};

template<typename T>
class cgpt_multi_arg_coarse_operator : public cgpt_multi_arg_fermion_operator_base {
public:
  T* op;

  cgpt_multi_arg_coarse_operator(T* _op) : op(_op) {
  }

  virtual ~cgpt_multi_arg_coarse_operator() {
    delete op;
  }

  virtual RealD unary(int opcode, cgpt_Lattice_base* in, cgpt_Lattice_base* out) {
    ASSERT(0);
    return 0.0;
  }

  virtual RealD dirdisp(int opcode, cgpt_Lattice_base* in, cgpt_Lattice_base* out, int dir, int disp) {
    ASSERT(0);
    return 0.0;
  }

  virtual RealD unary(int opcode,
                      std::vector<cgpt_Lattice_base*>& in, long in_n_virtual,
                      std::vector<cgpt_Lattice_base*>& out, long out_n_virtual) {
    return cgpt_multi_arg_fermion_operator_unary<T>(*op, opcode, in, in_n_virtual, out, out_n_virtual);
  }

  virtual RealD dirdisp(int opcode,
                        std::vector<cgpt_Lattice_base*>& in, long in_n_virtual,
                        std::vector<cgpt_Lattice_base*>& out, long out_n_virtual,
                        int dir, int disp) {
    return cgpt_multi_arg_fermion_operator_dirdisp<T>(*op, opcode,
                                                      in, in_n_virtual,
                                                      out, out_n_virtual,
                                                      dir, disp);
  }

  virtual void update(PyObject* args) {
    typedef typename T::VirtualLinkField::vector_type vCoeff_t;
    const int nbasis_virtual = GridTypeMapper<typename T::VirtualFermionField::vector_object>::count;

    std::cout << "nbasis_virtual in update: " << nbasis_virtual << std::endl;

    auto ASelfInv_ptr_list = get_pointer_vec<cgpt_Lattice_base>(args,"U_self_inv");
    auto A_ptr_list = get_pointer_vec<cgpt_Lattice_base>(args,"U");

    PVector<Lattice<iMatrix<iSinglet<vCoeff_t>,nbasis_virtual>>> ASelfInv;
    PVector<Lattice<iMatrix<iSinglet<vCoeff_t>,nbasis_virtual>>> A;

    cgpt_basis_fill(ASelfInv, ASelfInv_ptr_list);
    cgpt_basis_fill(A, A_ptr_list);

    ASSERT(A.size() == 9*ASelfInv.size());

    op->ImportGauge(A, ASelfInv); // TODO: rename -> ImportGauge for consistency?
  }
};
