/*
    GPT - Grid Python Toolkit
    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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
cgpt_fermion_operator_base* cgpt_create_wilson_twisted_mass(PyObject* args) {

  typedef WilsonImpl<vCoeff_t, FundamentalRepresentation, CoeffReal > WI;
  typedef typename vCoeff_t::scalar_type Coeff_t;

  typename WI::ImplParams wp;

  auto grid = get_pointer<GridCartesian>(args,"U_grid");
  auto grid_rb = get_pointer<GridRedBlackCartesian>(args,"U_grid_rb");
  RealD mass = get_float(args,"mass");
  RealD mu = get_float(args,"mu");

  wp.boundary_phases = get_complex_vec<Nd>(args,"boundary_phases");
  wp.overlapCommsCompute = true;

  Lattice< iLorentzColourMatrix< vCoeff_t > > U(grid);
  for (int mu=0;mu<Nd;mu++) {
    auto l = get_pointer<cgpt_Lattice_base>(args,"U",mu);
    auto& Umu = compatible<iColourMatrix<vCoeff_t>>(l)->l;
    PokeIndex<LorentzIndex>(U,Umu,mu);
  }

  auto f = new WilsonTMFermion<WI>(U, *grid, *grid_rb, mass, mu, wp);
  return new cgpt_fermion_operator<WilsonTMFermion<WI>>(f);
}
