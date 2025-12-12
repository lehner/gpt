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

template<typename WI> void cgpt_fermion_set_mass(WilsonFermion<WI>& op, PyObject* args) {
  RealD mass = get_float(args,"mass");
  op.mass = mass;
  if  (op.anisotropyCoeff.isAnisotropic){
    op.diag_mass = op.mass + 1.0 + (Nd-1)*(op.anisotropyCoeff.nu / op.anisotropyCoeff.xi_0);
  } else {
    op.diag_mass = 4.0 + op.mass;
  }
}

template<typename WI> void cgpt_fermion_set_mass(CompactWilsonClover5D<WI>& op, PyObject* args) {
  RealD mass = get_float(args,"mass");
  op.M5 = mass;
  //if (op.anisotropyCoeff.isAnisotropic){
  //  op.diag_mass = mass + 1.0 + (Nd-1)*(op.anisotropyCoeff.nu / op.anisotropyCoeff.xi_0);
  //} else {
  //  op.diag_mass = 4.0 + mass;
  //}
}

template<typename WI> void cgpt_fermion_set_mass(CayleyFermion5D<WI>& op, PyObject* args) {
  RealD mass_plus = get_float(args,"mass_plus");
  RealD mass_minus = get_float(args,"mass_minus");
  op.SetMass(mass_plus, mass_minus);
}

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

  virtual void set_mass(PyObject* args) {
    cgpt_fermion_set_mass(*op, args);
  }

};

template<typename T>
class cgpt_fermion_operator_with_vector_field : public cgpt_fermion_operator<T> {
public:
  cgpt_fermion_operator_with_vector_field(T* _op) : cgpt_fermion_operator<T>(_op) {
  }

  virtual void update(PyObject* args) {
    long num_gauge_fields = PyList_Size(get_key(args,"U"));
    if (num_gauge_fields == Nd*2)
      cgpt_create_aslashed(this->op->Aslashed, args);
    cgpt_fermion_operator<T>::update(args);
  }
};
