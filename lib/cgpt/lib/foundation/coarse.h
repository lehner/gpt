/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Daniel Richtmann (daniel.richtmann@ur.de, https://github.com/lehner/gpt)
                  2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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


    This code is based on original Grid code.
*/

template<class Field>
void conformable(GridBase* grid, const PVector<Field>& field) {
  for(int v=0; v<field.size(); v++) {
    conformable(grid, field[v].Grid());
  }
}


template<class Field>
void conformable(const PVector<Field>& lhs, const PVector<Field>& rhs) {
  assert(lhs.size() == rhs.size());
  for(int v=0; v<lhs.size(); v++) {
    conformable(lhs[v], rhs[v]);
  }
}


template<class Field>
void constantCheckerboard(const PVector<Field>& in, PVector<Field>& out) {
  assert(in.size() == out.size());
  for(int v=0; v<in.size(); v++) {
    out[v].Checkerboard() = in[v].Checkerboard();
  }
}


template<class Field>
void changingCheckerboard(const PVector<Field>& in, PVector<Field>& out) {
  assert(in.size() == out.size());
  for(int v=0; v<in.size(); v++) {
    if      (in[v].Checkerboard() == Even) out[v].Checkerboard() = Odd;
    else if (in[v].Checkerboard() == Odd)  out[v].Checkerboard() = Even;
    else assert(0);
  }
}


// this is the interface that gpt requires at the moment
template<class FermionField>
class MultiArgFermionOperatorBase {
public:
  /////////////////////////////////////////////////////////////////////////////
  //                                  grids                                  //
  /////////////////////////////////////////////////////////////////////////////

  virtual GridBase* FermionGrid()         = 0;
  virtual GridBase* FermionRedBlackGrid() = 0;
  virtual GridBase* GaugeGrid()           = 0;
  virtual GridBase* GaugeRedBlackGrid()   = 0;
  GridBase* Grid()         { return FermionGrid(); }
  GridBase* RedBlackGrid() { return FermionRedBlackGrid(); }

  /////////////////////////////////////////////////////////////////////////////
  //                         info about diagonal term                        //
  /////////////////////////////////////////////////////////////////////////////

  virtual int ConstEE()     = 0;
  virtual int isTrivialEE() = 0;

  /////////////////////////////////////////////////////////////////////////////
  //                            full cb operations                           //
  /////////////////////////////////////////////////////////////////////////////

  virtual void M(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void Mdag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void MdagM(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void Mdiag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void Mdir(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dir, int disp) = 0;
  virtual void MdirAll(const FermionField& in, uint64_t in_n_virtual, std::vector<FermionField>& out, uint64_t out_n_virtual) = 0;

  /////////////////////////////////////////////////////////////////////////////
  //                            half cb operations                           //
  /////////////////////////////////////////////////////////////////////////////

  virtual void Meooe(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void Mooee(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void MooeeInv(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void MeooeDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void MooeeDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void MooeeInvDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;

  /////////////////////////////////////////////////////////////////////////////
  //               non-hermitian hopping term; half cb or both               //
  /////////////////////////////////////////////////////////////////////////////

  virtual void Dhop(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) = 0;
  virtual void DhopOE(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) = 0;
  virtual void DhopEO(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) = 0;
  virtual void DhopDir(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dir, int disp) = 0;

  /////////////////////////////////////////////////////////////////////////////
  //                               dminus stuff                              //
  /////////////////////////////////////////////////////////////////////////////

  virtual void Dminus(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void DminusDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;

  /////////////////////////////////////////////////////////////////////////////
  //                              import/export                              //
  /////////////////////////////////////////////////////////////////////////////

  virtual void ImportPhysicalFermionSource(const FermionField& input, uint64_t input_n_virtual, FermionField& imported, uint64_t imported_n_virtual) = 0;
  virtual void ImportUnphysicalFermion(const FermionField& input, uint64_t input_n_virtual, FermionField& imported, uint64_t imported_n_virtual) = 0;
  virtual void ExportPhysicalFermionSolution(const FermionField& solution, uint64_t solution_n_virtual, FermionField& exported, uint64_t exported_n_virtual) = 0;
  virtual void ExportPhysicalFermionSource(const FermionField& solution, uint64_t solution_n_virtual, FermionField& exported, uint64_t exported_n_virtual) = 0;
};


template<class CComplex,int NbasisVirtual>
class MultiArgVirtualCoarsenedMatrix : public MultiArgFermionOperatorBase<PVector<Lattice<iVector<CComplex,NbasisVirtual>>>> {
public: // type definitions ///////////////////////////////////////////////////

  // site-wise types
  typedef         iVector<CComplex, NbasisVirtual>          SiteSpinor;
  typedef         iMatrix<CComplex, NbasisVirtual>          SiteMatrix;
  typedef iVector<iMatrix<CComplex, NbasisVirtual>, 2*Nd+1> DoubleStoredSiteMatrix;

  // lattice types = virtual fields
  typedef Lattice<SiteSpinor>             VirtualFermionField;
  typedef Lattice<SiteMatrix>             VirtualLinkField;
  typedef Lattice<SiteMatrix>             VirtualGridLayoutGaugeField;
  typedef Lattice<DoubleStoredSiteMatrix> VirtualDoubleStoredGaugeField;

  // physical fields, used internally
  typedef std::vector<VirtualFermionField>           PhysicalFermionField;
  typedef std::vector<VirtualLinkField>              PhysicalLinkField;
  typedef std::vector<VirtualGridLayoutGaugeField>   PhysicalGridLayoutGaugeField;
  typedef std::vector<VirtualDoubleStoredGaugeField> PhysicalGaugeField;

  // used by the outside world
  typedef PVector<VirtualFermionField>                FermionField;
  typedef PVector<VirtualLinkField>                   LinkField;
  typedef PVector<VirtualGridLayoutGaugeField>        GridLayoutGaugeField;
  typedef CartesianStencil<SiteSpinor,SiteSpinor,int> Stencil;
  typedef typename SiteSpinor::vector_type            vCoeff_t;

private: // member data ///////////////////////////////////////////////////////

  Geometry geom_;
  Geometry geomMultiArg_;

  GridBase* grid_;
  GridBase* cbGrid_;
  GridBase* gridMultiArg_;
  GridBase* cbGridMultiArg_;

  bool hermitianOverall_;
  bool hermitianSelf_;

  uint64_t link_n_virtual_;
  uint64_t fermion_n_virtual_;
  uint64_t n_arg_;

  Stencil stencil_;
  // Stencil stencilEven_;
  // Stencil stencilOdd_;
  Stencil stencilMultiArg_;
  // Stencil stencilEvenMultiArg_;
  // Stencil stencilOddMultiArg_;

  PhysicalGaugeField Uc_;
  // PhysicalGaugeField UcEven_;
  // PhysicalGaugeField UcOdd_;

  PhysicalGridLayoutGaugeField UcGridLayout_;

  PhysicalLinkField UcSelfInv_;
  // PhysicalLinkField UcSelfInvEven_;
  // PhysicalLinkField UcSelfInvOdd_;

  VirtualFermionField tmpMultiArg_;

  double MCalls;
  double MMiscTime;
  double MViewTime;
  double MView2Time;
  double MCopyTime;
  double MCommTime;
  double MComputeTime;
  double MTotalTime;

  // Vector<RealD> dagFactor_;

public: // member functions (implementing interface) //////////////////////////

  // grids
  GridBase* FermionGrid()         { return grid_; }
  GridBase* FermionRedBlackGrid() { return cbGrid_; }
  GridBase* GaugeGrid()           { return grid_; }
  GridBase* GaugeRedBlackGrid()   { return cbGrid_; }

  // info about diagonal term
  int ConstEE()     { return 0; }
  int isTrivialEE() { return 0; }

  // full cb operations
  void M(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    M_internal(in, in_n_virtual, out, out_n_virtual);
  }
  void Mdag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void MdagM(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void Mdiag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void Mdir(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dir, int disp) {}
  void MdirAll(const FermionField& in, uint64_t in_n_virtual, std::vector<FermionField>& out, uint64_t out_n_virtual) {}

  // half cb operations
  void Meooe(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void Mooee(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void MooeeInv(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void MeooeDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void MooeeDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void MooeeInvDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}

  // non-hermitian hopping term; half cb or both
  void Dhop(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) {}
  void DhopOE(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) {}
  void DhopEO(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) {}
  void DhopDir(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dir, int disp) {}

  // dminus stuff

  void Dminus(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void DminusDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}

  // import/export
  void ImportPhysicalFermionSource(const FermionField& input, uint64_t input_n_virtual, FermionField& imported, uint64_t imported_n_virtual) {}
  void ImportUnphysicalFermion(const FermionField& input, uint64_t input_n_virtual, FermionField& imported, uint64_t imported_n_virtual) {}
  void ExportPhysicalFermionSolution(const FermionField& solution, uint64_t solution_n_virtual, FermionField& exported, uint64_t exported_n_virtual) {}
  void ExportPhysicalFermionSource(const FermionField& solution, uint64_t solution_n_virtual, FermionField& exported, uint64_t exported_n_virtual) {}

public: // member functions (additional) //////////////////////////////////////

  // helpers
  void ImportGauge(const PVector<VirtualLinkField>& Uc, const PVector<VirtualLinkField>& UcSelfInv) {
    assert(Uc.size() == geom_.npoint * Uc_.size());
    assert(UcSelfInv.size() == UcSelfInv_.size());

    conformable(GaugeGrid(), Uc);
    conformable(GaugeGrid(), UcSelfInv);

    const int Nsite           = GaugeGrid()->oSites();
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Npoint          = geom_.npoint;
    const int NvirtualLink    = link_n_virtual_;
    const int NvirtualFermion = fermion_n_virtual_;

    // NOTE: can't use PokeIndex here because of different tensor depths
    VECTOR_VIEW_OPEN_POINTER(Uc_,           Uc_member_v,             Uc_member_p,             AcceleratorWrite);
    VECTOR_VIEW_OPEN_POINTER(UcGridLayout_, Uc_member_grid_layout_v, Uc_member_grid_layout_p, AcceleratorWrite);
    VECTOR_VIEW_OPEN_POINTER(Uc,            Uc_arg_v,                Uc_arg_p,                AcceleratorRead);
    for(int v=0; v<NvirtualLink; v++) {
      const int v_row = v%NvirtualFermion; const int v_col = v/NvirtualFermion; // NOTE: comes in from gpt with row faster index -> col-major order
      const int link_idx_row_col = v_row * NvirtualFermion + v_col;
      const int link_idx_col_row = v_col * NvirtualFermion + v_row;
      for(int p=0; p<Npoint; p++) {
        const int gauge_idx_point_row_col = p * NvirtualLink + v_row * NvirtualFermion + v_col;
        const int gauge_idx_point_col_row = p * NvirtualLink + v_col * NvirtualFermion + v_row;
        const int gauge_idx_row_col_point = v_row * NvirtualFermion * Npoint + v_col * Npoint + p;
        const int gauge_idx_col_row_point = v_col * NvirtualFermion * Npoint + v_row * Npoint + p;

        accelerator_for(ss, Nsite, Nsimd, {
          // new layout with Lorentz in tensor
          coalescedWrite(Uc_member_p[link_idx_row_col][ss](p), coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss])); // change to col faster index -> row-major order -> with transpose
          // coalescedWrite(Uc_member_p[link_idx_col_row][ss](p), coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss])); // keep      row faster index -> col-major order -> without transpose

          // grid's layout with Lorentz in std::vector
          // coalescedWrite(Uc_member_grid_layout_p[gauge_idx_point_row_col][ss], coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss])); // point slow, virtual fast, col faster index -> virtual in row-major order
          // coalescedWrite(Uc_member_grid_layout_p[gauge_idx_point_col_row][ss], coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss])); // point slow, virtual fast, row faster index -> virtual in col-major order
          coalescedWrite(Uc_member_grid_layout_p[gauge_idx_row_col_point][ss], coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss])); // virtual slow, point fast, col faster index -> virtual in row-major order
          // coalescedWrite(Uc_member_grid_layout_p[gauge_idx_col_row_point][ss], coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss])); // virtual slow, point fast, row faster index -> virtual in col-major order
        });
      }
    }
    VECTOR_VIEW_CLOSE_POINTER(Uc_member_v, Uc_member_p);
    VECTOR_VIEW_CLOSE_POINTER(Uc_member_grid_layout_v, Uc_member_grid_layout_p);
    VECTOR_VIEW_CLOSE_POINTER(Uc_arg_v, Uc_arg_p);

    for(int v=0; v<NvirtualLink; v++) UcSelfInv_[v] = UcSelfInv[v];
    grid_printf("ImportGauge of new Coarse Operator finished\n");
  }

  void PickCheckerboards() {
    grid_printf("VirtualCoarsenedMatrix::pickCheckerboards still needs to be implemented\n");
  }

  void Report(int Nvec) {
    assert(Nvec == n_arg_);
    RealD Nproc = grid_->_Nprocessors;
    RealD Nnode = grid_->NodeCount();
    RealD volume = 1;
    Coordinate latt = grid_->GlobalDimensions();
    for(int mu=0;mu<Nd;mu++) volume=volume*latt[mu];
    RealD nbasis = NbasisVirtual * fermion_n_virtual_;

    if ( MCalls > 0 ) {
      grid_printf("#### M calls report\n");
      grid_printf("CoarseOperator Number of Calls                         : %d\n", (int)MCalls);
      grid_printf("CoarseOperator MiscTime   /Calls, MiscTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MMiscTime   /MCalls, MMiscTime,    MMiscTime   /MTotalTime*100);
      grid_printf("CoarseOperator ViewTime   /Calls, ViewTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MViewTime   /MCalls, MViewTime,    MViewTime   /MTotalTime*100);
      grid_printf("CoarseOperator View2Time  /Calls, View2Time   : %10.2f us, %10.2f us (= %6.2f %%)\n", MView2Time  /MCalls, MView2Time,   MView2Time  /MTotalTime*100);
      grid_printf("CoarseOperator CopyTime   /Calls, CopyTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MCopyTime   /MCalls, MCopyTime,    MCopyTime   /MTotalTime*100);
      grid_printf("CoarseOperator CommTime   /Calls, CommTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MCommTime   /MCalls, MCommTime,    MCommTime   /MTotalTime*100);
      grid_printf("CoarseOperator ComputeTime/Calls, ComputeTime : %10.2f us, %10.2f us (= %6.2f %%)\n", MComputeTime/MCalls, MComputeTime, MComputeTime/MTotalTime*100);
      grid_printf("CoarseOperator TotalTime  /Calls, TotalTime   : %10.2f us, %10.2f us (= %6.2f %%)\n", MTotalTime  /MCalls, MTotalTime,   MTotalTime  /MTotalTime*100);

      // Average the compute time
      grid_->GlobalSum(MComputeTime);
      MComputeTime/=Nproc;
      RealD complex_words = 2;
      RealD prec_bytes    = getPrecision<typename CComplex::vector_type>::value * 4; // 4 for float, 8 for double
      RealD flop_per_site = 1.0 * (2 * nbasis * (36 * nbasis - 1)) * Nvec;
      RealD word_per_site = 1.0 * (9 * nbasis + 9 * nbasis * nbasis + nbasis) * Nvec;
      RealD byte_per_site = word_per_site * complex_words * prec_bytes;
      RealD mflops = flop_per_site*volume*MCalls/MComputeTime;
      RealD mbytes = byte_per_site*volume*MCalls/MComputeTime;
      grid_printf("CoarseOperator Average mflops/s, mbytes/s per call                : %.0f, %.0f\n", mflops, mbytes);
      grid_printf("CoarseOperator Average mflops/s, mbytes/s per call per rank       : %.0f, %.0f\n", mflops/Nproc, mbytes/Nproc);
      grid_printf("CoarseOperator Average mflops/s, mbytes/s per call per node       : %.0f, %.0f\n", mflops/Nnode, mbytes/Nnode);

      RealD Fullmflops = flop_per_site*volume*MCalls/(MTotalTime);
      RealD Fullmbytes = byte_per_site*volume*MCalls/(MTotalTime);
      grid_printf("CoarseOperator Average mflops/s, mbytes/s per call (full)         : %.0f, %.0f\n", Fullmflops, Fullmbytes);
      grid_printf("CoarseOperator Average mflops/s, mbytes/s per call per rank (full): %.0f, %.0f\n", Fullmflops/Nproc, Fullmbytes/Nproc);
      grid_printf("CoarseOperator Average mflops/s, mbytes/s per call per node (full): %.0f, %.0f\n", Fullmflops/Nnode, Fullmbytes/Nnode);

      grid_printf("CoarseOperator Stencil\n"); stencil_.Report();
      // grid_printf("CoarseOperator StencilEven\n"); stencilEven_.Report();
      // grid_printf("CoarseOperator StencilOdd\n"); stencilOdd_.Report();
      grid_printf("CoarseOperator StencilMultiArg\n"); stencilMultiArg_.Report();
      // grid_printf("CoarseOperator StencilMultiArgEven\n"); stencilMultiArgEven_.Report();
      // grid_printf("CoarseOperator StencilMultiArgOdd\n"); stencilMultiArgOdd_.Report();
    }
    grid_printf("Report of new Coarse Operator finished\n");
  }

  void ZeroCounters() {
    MCalls       = 0; // ok
    MMiscTime    = 0;
    MViewTime    = 0;
    MView2Time   = 0;
    MCopyTime    = 0;
    MCommTime    = 0;
    MComputeTime = 0;
    MTotalTime   = 0;

    stencil_.ZeroCounters();
    // stencilEven_.ZeroCounters();
    // stencilOdd_.ZeroCounters();
    stencilMultiArg_.ZeroCounters();
    // stencilMultiArgEven_.ZeroCounters();
    // stencilMultiArgOdd_.ZeroCounters();
  }

  // constructors
  MultiArgVirtualCoarsenedMatrix(const PVector<VirtualLinkField>& Uc,
                                 const PVector<VirtualLinkField>& UcSelfInv,
                                 GridCartesian&                 grid,
                                 GridRedBlackCartesian&         rbGrid,
                                 int                            makeHermitian,
                                 int                            numArg)
    : geom_(grid._ndimension)
    , geomMultiArg_(grid._ndimension+1)
    , grid_(&grid)
    , cbGrid_(&rbGrid)
    , gridMultiArg_(SpaceTimeGrid::makeFiveDimGrid(uint64_t(sqrt(UcSelfInv.size())), &grid))
    , cbGridMultiArg_(SpaceTimeGrid::makeFiveDimRedBlackGrid(uint64_t(sqrt(UcSelfInv.size())), &grid))
    , hermitianOverall_(makeHermitian)
    , hermitianSelf_(false)
    , link_n_virtual_(UcSelfInv.size())
    , fermion_n_virtual_(uint64_t(sqrt(UcSelfInv.size())))
    , n_arg_(numArg)
    , stencil_(grid_, geom_.npoint, Even, geom_.directions, geom_.displacements, 0)
    // , stencilEven_(cbGrid_, geom_.npoint, Even, geom_.directions, geom_.displacements, 0)
    // , stencilOdd_(cbGrid_, geom_.npoint, Odd, geom_.directions, geom_.displacements, 0)
    , stencilMultiArg_(gridMultiArg_, geomMultiArg_.npoint, Even, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
    // , stencilEvenMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint, Even, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
    // , stencilOddMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint, Odd, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
    , Uc_(link_n_virtual_, grid_)
    // , UcEven_(link_n_virtual_, cbGrid_)
    // , UcOdd_(link_n_virtual_, cbGrid_)
    , UcGridLayout_(geom_.npoint*link_n_virtual_, grid_)
    , UcSelfInv_(link_n_virtual_, grid_)
    // , UcSelfInvEven_(link_n_virtual_, cbGrid_)
    // , UcSelfInvOdd_(link_n_virtual_, cbGrid_)
    , tmpMultiArg_(gridMultiArg_)
  {
    grid_->show_decomposition();
    cbGrid_->show_decomposition();
    gridMultiArg_->show_decomposition();
    cbGridMultiArg_->show_decomposition();
    ImportGauge(Uc, UcSelfInv);
    PickCheckerboards();
    ZeroCounters();
    grid_printf("Constructed the latest coarse operator\n"); fflush(stdout);
  }

private: // member functions //////////////////////////////////////////////////

  int numArg(const FermionField& a, uint64_t a_n_virtual, const FermionField& b, uint64_t b_n_virtual) const {
    int a_size = a.size();
    int b_size = b.size();
    assert(a_size == b_size);
    assert(a_n_virtual == b_n_virtual);
    assert(a_n_virtual == fermion_n_virtual_);
    assert(a_size >= a_n_virtual);
    assert(a_size % a_n_virtual == 0);
    return a_size / a_n_virtual;
  }

public: // kernel functions TODO: move somewhere else ////////////////////////

  void M_internal(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
#if defined(GRID_CUDA) || defined(GRID_HIP)
    M_gpu(in, in_n_virtual, out, out_n_virtual);
#else
    M_cpu(in, in_n_virtual, out, out_n_virtual);
#endif
  }

  void M_gpu(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: corresponds to "M_finegrained_loopinternal_tensorlayout_parchange_commsreduce" in test file
    // NOTE: version with additional parallelism over output virtual index + reducing comms by temporary 5d object -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = this->Grid()->oSites();
    const int Npoint          = geom_.npoint;

    assert(n_arg_ == Narg);

    conformable(FermionGrid(), in);
    conformable(FermionGrid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    {
      MView2Time-=usecond();
      VECTOR_VIEW_OPEN_POINTER(in, in_v, in_p, AcceleratorRead);
      autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorWrite);
      MView2Time+=usecond();
      MCopyTime-=usecond();
      accelerator_for(sF, Nsite*NvirtualFermion*Narg, Nsimd, {
              int _sF   = sF;                  // this does fastest to slowest from top to bottom
        const int arg   = _sF%Narg;            _sF/=Narg;
        const int v_col = _sF%NvirtualFermion; _sF/=NvirtualFermion;
        const int sU    = _sF%Nsite;           _sF/=Nsite;
        coalescedWrite(tmpMultiArg_v[sF], in_v[arg*NvirtualFermion+v_col](sU));
        // printf("COPY: sF = %4d, arg = %4d, sU = %4d, v_col = %4d\n", sF, arg, sU, v_col); fflush(stdout);
      });
      MCopyTime+=usecond();
      MView2Time-=usecond();
      VECTOR_VIEW_CLOSE_POINTER(in_v, in_p);
      MView2Time+=usecond();
    }

    MCommTime-=usecond();
    stencilMultiArg_.HaloExchange(tmpMultiArg_, compressor);
    MCommTime+=usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_v, Uc_p, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(in, in_v, in_p, AcceleratorRead);
    autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorRead);
    autoView(stencilMultiArg_v, stencilMultiArg_, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmpMultiArg_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmpMultiArg_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_for(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
              int _idx  = idx;
        const int b     = _idx%NbasisVirtual; _idx/=NbasisVirtual;
        const int arg   = _idx%Narg; _idx/=Narg;
        const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
        const int ss    = _idx%Nsite; _idx/=Nsite;

        const int v_arg_col = arg*NvirtualFermion+v_col;
        const int v_arg_row = arg*NvirtualFermion+v_row;
        const int v_row_col = v_row * NvirtualFermion + v_col;
        const int v_col_row = v_col * NvirtualFermion + v_row;
        const int sF        = ss*NvirtualFermion*Narg+v_col*Narg+arg; // needed for stencil access

        calcComplex res;
        calcVector nbr;
        int ptype;
        StencilEntry *SE_MA;

        if (v_col == 0)
          res = Zero();
        else
          res = coalescedRead(out_p[v_arg_row][ss](b));

        for(int point=0; point<Npoint; point++) {
          SE_MA=stencilMultiArg_v.GetEntry(ptype,point,sF);

          if(SE_MA->_is_local) {
            nbr = coalescedReadPermute(tmpMultiArg_v[SE_MA->_offset],ptype,SE_MA->_permute);
          } else {
            nbr = coalescedRead(stencilMultiArg_v.CommBuf()[SE_MA->_offset]);
          }
          acceleratorSynchronise();

          for(int bb=0;bb<NbasisVirtual;bb++) {
            res = res + coalescedRead(Uc_p[v_row*NvirtualFermion+v_col][ss](point)(b,bb))*nbr(bb);
            // res = res + coalescedRead(Uc_p[v_col*NvirtualFermion+v_row][ss](point)(b,bb))*nbr(bb);
          }
        }
        coalescedWrite(out_p[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    VECTOR_VIEW_CLOSE_POINTER(in_v, in_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
  }

  void M_cpu(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: corresponds to "M_loopinternal_tensorlayout_parchange_commsreduce" in test file
    // NOTE: version with additional parallelism over output virtual index + reducing comms by temporary 5d object -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = this->Grid()->oSites();
    const int Npoint          = geom_.npoint;

    assert(n_arg_ == Narg);

    conformable(FermionGrid(), in);
    conformable(FermionGrid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    {
      MView2Time-=usecond();
      VECTOR_VIEW_OPEN_POINTER(in, in_v, in_p, AcceleratorRead);
      autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorWrite);
      MView2Time+=usecond();
      MCopyTime-=usecond();
      accelerator_for(sF, Nsite*NvirtualFermion*Narg, Nsimd, {
              int _sF   = sF;                  // this does fastest to slowest from top to bottom
        const int arg   = _sF%Narg;            _sF/=Narg;
        const int v_col = _sF%NvirtualFermion; _sF/=NvirtualFermion;
        const int sU    = _sF%Nsite;           _sF/=Nsite;
        coalescedWrite(tmpMultiArg_v[sF], in_v[arg*NvirtualFermion+v_col](sU));
        // printf("COPY: sF = %4d, arg = %4d, sU = %4d, v_col = %4d\n", sF, arg, sU, v_col); fflush(stdout);
      });
      MCopyTime+=usecond();
      MView2Time-=usecond();
      VECTOR_VIEW_CLOSE_POINTER(in_v, in_p);
      MView2Time+=usecond();
    }

    MCommTime-=usecond();
    stencilMultiArg_.HaloExchange(tmpMultiArg_, compressor);
    MCommTime+=usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_v, Uc_p, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(in, in_v, in_p, AcceleratorRead);
    autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorRead);
    autoView(stencilMultiArg_v, stencilMultiArg_, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmpMultiArg_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmpMultiArg_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_for(idx, Nsite*NvirtualFermion*Narg, Nsimd, {
              int _idx  = idx;
        const int arg   = _idx%Narg; _idx/=Narg;
        const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
        const int ss    = _idx%Nsite; _idx/=Nsite;

        const int v_arg_col = arg*NvirtualFermion+v_col;
        const int v_arg_row = arg*NvirtualFermion+v_row;
        const int v_row_col = v_row * NvirtualFermion + v_col;
        const int v_col_row = v_col * NvirtualFermion + v_row;
        const int sF        = ss*NvirtualFermion*Narg+v_col*Narg+arg; // needed for stencil access

        calcVector res;
        calcVector nbr;
        int ptype;
        StencilEntry *SE_MA;

        if (v_col == 0)
          res = Zero();
        else
          res = coalescedRead(out_p[v_arg_row][ss]);

        for(int point=0; point<Npoint; point++) {
          SE_MA=stencilMultiArg_v.GetEntry(ptype,point,sF);

          if(SE_MA->_is_local) {
            nbr = coalescedReadPermute(tmpMultiArg_v[SE_MA->_offset],ptype,SE_MA->_permute);
          } else {
            nbr = coalescedRead(stencilMultiArg_v.CommBuf()[SE_MA->_offset]);
          }
          acceleratorSynchronise();

          res = res + coalescedRead(Uc_p[v_row*NvirtualFermion+v_col][ss](point))*nbr;
          // res = res + coalescedRead(Uc_p[v_col*NvirtualFermion+v_row][ss](point))*nbr;
        }
        coalescedWrite(out_p[v_arg_row][ss],res);
      });
      MComputeTime += usecond();
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    VECTOR_VIEW_CLOSE_POINTER(in_v, in_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
  }
};
