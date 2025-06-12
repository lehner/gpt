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
cgpt_fermion_operator_base* cgpt_create_wilson_clover(PyObject* args) {

  typedef WilsonImpl<vCoeff_t, FundamentalRepresentation, CoeffReal > WI;
  typedef typename vCoeff_t::scalar_type Coeff_t;

  WilsonAnisotropyCoefficients wac;
  typename WI::ImplParams wp;

  auto grid = get_pointer<GridCartesian>(args,"U_grid");
  auto grid_rb = get_pointer<GridRedBlackCartesian>(args,"U_grid_rb");
  RealD mass = get_float(args,"mass");
  RealD csw_r = get_float(args,"csw_r");
  RealD csw_t = get_float(args,"csw_t");
  RealD cF = get_float(args,"cF");
  bool multi_rhs = get_bool(args,"multi_rhs");
  bool use_legacy = get_bool(args,"use_legacy");
  wac.isAnisotropic = get_bool(args,"isAnisotropic");
  wac.xi_0 = get_float(args,"xi_0");
  wac.nu = get_float(args,"nu");
  wp.boundary_phases = get_complex_vec<Nd>(args,"boundary_phases");

  // Q =   1 + (Nd-1)/xi_0 + m
  //     + W_t + (nu/xi_0) * W_s
  //     - 1/2*[ csw_t * sum_s (sigma_ts F_ts) + (csw_s/xi_0) * sum_ss (sigma_ss F_ss)  ]

  wp.overlapCommsCompute = true;

  Lattice< iLorentzColourMatrix< vCoeff_t > > U(grid);
  for (int mu=0;mu<Nd;mu++) {
    auto l = get_pointer<cgpt_Lattice_base>(args,"U",mu);
    auto& Umu = compatible<iColourMatrix<vCoeff_t>>(l)->l;
    PokeIndex<LorentzIndex>(U,Umu,mu);
  }

  if (!use_legacy) {
    if (!multi_rhs) {
      auto f = new CompactWilsonClover<WI>(U,*grid,*grid_rb,mass,csw_r,csw_t,cF,wac,wp);
      return new cgpt_fermion_operator<CompactWilsonClover<WI>>(f);
    } else {
      auto grid5d = get_pointer<GridCartesian>(args,"F_grid");
      auto grid5d_rb = get_pointer<GridRedBlackCartesian>(args,"F_grid_rb");

      ASSERT(!wac.isAnisotropic && wac.xi_0 == 1.0 && wac.nu == 1.0);
      
      auto f = new CompactWilsonClover5D<WI>(U,*grid5d,*grid5d_rb,*grid,*grid_rb,mass,csw_r,csw_t,cF,wp);
      return new cgpt_fermion_operator<CompactWilsonClover5D<WI>>(f);
    }
  } else { // TODO: deprecate soon
    auto f = new WilsonClover<WI>(U, *grid, *grid_rb, mass, csw_r, csw_t, wac, wp);
    return new cgpt_fermion_operator<WilsonClover<WI>>(f);
  }
}
