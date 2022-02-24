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
template<typename vCoeff_t>
cgpt_fermion_operator_base* cgpt_create_zmobius(PyObject* args) {

  typedef WilsonImpl<vCoeff_t, FundamentalRepresentation, CoeffComplex > WI;
  typedef typename vCoeff_t::scalar_type Coeff_t;

  typename WI::ImplParams wp;

  auto grid5 = get_pointer<GridCartesian>(args,"F_grid");
  auto grid5_rb = get_pointer<GridRedBlackCartesian>(args,"F_grid_rb");
  auto grid4 = get_pointer<GridCartesian>(args,"U_grid");
  auto grid4_rb = get_pointer<GridRedBlackCartesian>(args,"U_grid_rb");
  RealD mass_plus = get_float(args,"mass_plus");
  RealD mass_minus = get_float(args,"mass_minus");
  RealD M5 = get_float(args,"M5");
  RealD b = get_float(args,"b");
  RealD c = get_float(args,"c");
  std::vector<ComplexD> omega = get_complex_vec_gen(args,"omega");
  wp.boundary_phases = get_complex_vec<Nd>(args,"boundary_phases");
  wp.overlapCommsCompute = true;

  Lattice< iLorentzColourMatrix< vCoeff_t > > U(grid4);
  for (int mu=0;mu<Nd;mu++) {
    auto l = get_pointer<cgpt_Lattice_base>(args,"U",mu);
    auto& Umu = compatible<iColourMatrix<vCoeff_t>>(l)->l;
    PokeIndex<LorentzIndex>(U,Umu,mu);
  }

  auto f = new ZMobiusFermion<WI>(U,*grid5,*grid5_rb,*grid4,*grid4_rb,
				  mass_plus,M5,omega,b,c,wp);

  if (mass_plus != mass_minus)
    f->SetMass(mass_plus, mass_minus);

  return new cgpt_fermion_operator<ZMobiusFermion<WI>>(f);
}
