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


template<class CComplex,int Nbasis>
class MultiArgVirtualCoarsenedMatrix : public MultiArgFermionOperatorBase<PVector<Lattice<iVector<CComplex, Nbasis>>>> {
  /////////////////////////////////////////////////////////////////////////////
  //                             Type definitions                            //
  /////////////////////////////////////////////////////////////////////////////

public:

  // site-wise types
  typedef         iVector<CComplex, Nbasis>          SiteSpinor;
  typedef         iMatrix<CComplex, Nbasis>          SiteMatrix;
  typedef iVector<iMatrix<CComplex, Nbasis>, 2*Nd+1> DoubleStoredSiteMatrix;

  // lattice types
  typedef Lattice<SiteSpinor>             BasicFermionField;
  typedef Lattice<SiteMatrix>             BasicLinkField;
  typedef Lattice<DoubleStoredSiteMatrix> BasicDoubleStoredGaugeField;

  // to be used by the outside world
  typedef PVector<BasicFermionField>                  FermionField;
  typedef PVector<BasicLinkField>                     LinkField;
  typedef PVector<BasicLinkField>                     GaugeField;
  typedef CartesianStencil<SiteSpinor,SiteSpinor,int> Stencil;
  typedef typename SiteSpinor::vector_type            vCoeff_t;

  // data types used internally
  typedef std::vector<BasicFermionField>           InternalFermionField;
  typedef std::vector<BasicLinkField>              InternalLinkField;
  typedef std::vector<BasicDoubleStoredGaugeField> InternalGaugeField;

  /////////////////////////////////////////////////////////////////////////////
  //                               Member Data                               //
  /////////////////////////////////////////////////////////////////////////////

private:

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

  Stencil stencil_;
  Stencil stencilEven_;
  Stencil stencilOdd_;
  Stencil stencilMultiArg_;
  Stencil stencilEvenMultiArg_;
  Stencil stencilOddMultiArg_;

  InternalGaugeField Uc_;
  InternalGaugeField UcEven_;
  InternalGaugeField UcOdd_;

  InternalLinkField UcSelfInv_;
  InternalLinkField UcSelfInvEven_;
  InternalLinkField UcSelfInvOdd_;

  BasicFermionField tmpMultiArg_;

  double MCalls;
  double MMiscTime;
  double MViewTime;
  double MView2Time;
  double MCopyTime;
  double MCommTime;
  double MComputeTime;
  double MTotalTime;

  // Vector<RealD> dagFactor_;

  /////////////////////////////////////////////////////////////////////////////
  //                Member Functions (implementing interface)                //
  /////////////////////////////////////////////////////////////////////////////

public:

  // grids ////////////////////////////////////////////////////////////////////

  GridBase* Grid()                { return FermionGrid(); }
  GridBase* RedBlackGrid()        { return FermionRedBlackGrid(); }
  GridBase* FermionGrid()         { return grid_; }
  GridBase* FermionRedBlackGrid() { return cbGrid_; }
  GridBase* GaugeGrid()           { return grid_; }
  GridBase* GaugeRedBlackGrid()   { return cbGrid_; }

  // info about diagonal term /////////////////////////////////////////////////

  int ConstEE()     { return 0; }
  int isTrivialEE() { return 0; }

  // full cb operations ///////////////////////////////////////////////////////

  void M(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    MCalls++;
    MTotalTime-=usecond();
    MMiscTime-=usecond();
    conformable(this->Grid(), in);
    conformable(this->Grid(), out);
    constantCheckerboard(in, out);

    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = fermion_n_virtual_;
    const int NvirtualLink    = link_n_virtual_;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = this->Grid()->oSites();
    const int Npoint          = geom_.npoint;

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime+=usecond();

    // NOTE: using '_p' instead of '_v' gave a noticeable performance increase for small lattices
    MViewTime-=usecond();
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_v,  Uc_p,  AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorRead);
    MViewTime+=usecond();

    // NOTE: further improvements possible here:
    // - offload loop over arg -> save bandwidth for gauge
    // - perform halo exchanges for all virtual fields at once
    // for both: need to do all comms beforehand
    // but:      can't do vector<Stencil> because of "global" comms buffer per rank
    // -> need multi arg stencil

    ASSERT(Narg == 1); // the way it is done atm, it only works for 1 rhs!

    {
      MView2Time-=usecond();
      VECTOR_VIEW_OPEN_POINTER(in, in_v, in_p, AcceleratorRead);
      autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorWrite);
      MView2Time+=usecond();
      MCopyTime-=usecond();
      accelerator_for(sF, NvirtualFermion*Nsite*Narg, Nsimd, {
              int _sF   = sF;                  // this does fastest to slowest from top to bottom
        const int arg   = _sF%Narg;            _sF/=Narg;
        const int v_col = _sF%NvirtualFermion; _sF/=NvirtualFermion;
        const int sU    = _sF%Nsite;           _sF/=Nsite;
        coalescedWrite(tmpMultiArg_v[sF], in_v[arg*NvirtualFermion+v_col](sU));
        // printf("COPY: sF = %4d, arg = %4d, sU = %4d, v_col = %4d\n", sF, arg, sU, v_col); fflush(stdout);
      });
      MCopyTime+=usecond();
      MView2Time-=usecond();
      VECTOR_VIEW_CLOSE(in_v);
      MView2Time+=usecond();
    }

    MCommTime-=usecond();
    // stencil_.HaloExchange(in[arg_v_col],compressor);
    stencilMultiArg_.HaloExchange(tmpMultiArg_, compressor);
    MCommTime+=usecond();

    MViewTime-=usecond();
    // autoView(in_v, in[arg_v_col], AcceleratorRead);
    // autoView(stencil_v, stencil_, AcceleratorRead);
    autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorRead);
    autoView(stencilMultiArg_v, stencilMultiArg_, AcceleratorRead);
    MViewTime+=usecond();

    // for(int arg=0; arg<Narg; arg++) {
    // for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmpMultiArg_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmpMultiArg_v[0](0))) calcComplex;

      MComputeTime-=usecond();
      accelerator_for(sss, NvirtualFermion*Nsite*Narg*Nbasis, Nsimd, {
              int sss_      = sss;
        const int b_row     = sss_%Nbasis;          sss_/=Nbasis;
        const int arg       = sss_%Narg;            sss_/=Narg;
        const int ss        = sss_%Nsite;           sss_/=Nsite;
        const int v_row     = sss_%NvirtualFermion; sss_/=NvirtualFermion;
        const int arg_v_row = arg*NvirtualFermion+v_row;

        calcComplex res;
        calcVector nbr;
        int ptype;
        StencilEntry *SE;

        for(int v_col=0; v_col<NvirtualFermion; v_col++) {

        // const int arg_v_col = arg*NvirtualFermion+v_col;
        // const int v_link    = v_col * NvirtualFermion + v_row;
          const int v_link    = v_row * NvirtualFermion + v_col;
        const int sF        = ss*NvirtualFermion*Narg+v_col*Narg+arg; // needed for stencil access

        res = Zero();

        for(int point=0; point<Npoint; point++) {
          SE=stencilMultiArg_v.GetEntry(ptype,point,sF);
        // printf("MAIN: sss = %4d, b_row = %4d, arg = %4d, ss = %4d, v_row = %4d, arg_v_row = %4d, v_col = %4d, v_link = %4d, sF = %4d, point = %d, SE_offset = %4d\n",
        //        sss, b_row, arg, ss, v_row, arg_v_row, v_col, v_link, sF, point, SE->_offset
        //        ); fflush(stdout);
          if(SE->_is_local) {
            nbr = coalescedReadPermute(tmpMultiArg_v[SE->_offset],ptype,SE->_permute);
          } else {
            nbr = coalescedRead(stencilMultiArg_v.CommBuf()[SE->_offset]);
          }
          acceleratorSynchronise();
          for(int b_col=0; b_col<Nbasis; b_col++) {
            res = res + coalescedRead(Uc_p[v_link][ss](point)(b_row,b_col))*nbr(b_col);
          }
        }

        if(v_col == 0) {
          coalescedWrite(out_p[arg_v_row][ss](b_row), res);
        } else {
          auto out_acc = coalescedRead(out_p[arg_v_row][ss](b_row));
          coalescedWrite(out_p[arg_v_row][ss](b_row), out_acc+res);
        }
        }
      });
      MComputeTime+=usecond();
    // }
    // }

    MViewTime-=usecond();
    VECTOR_VIEW_CLOSE(Uc_v);
    VECTOR_VIEW_CLOSE(out_v);
    MViewTime+=usecond();
    MTotalTime+=usecond();
  }
  void Mdag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void MdagM(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void Mdiag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void Mdir(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dir, int disp) {}
  void MdirAll(const FermionField& in, uint64_t in_n_virtual, std::vector<FermionField>& out, uint64_t out_n_virtual) {}

  // half cb operations ///////////////////////////////////////////////////////

  void Meooe(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void Mooee(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void MooeeInv(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void MeooeDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void MooeeDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void MooeeInvDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}

  // non-hermitian hopping term; half cb or both //////////////////////////////

  void Dhop(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) {}
  void DhopOE(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) {}
  void DhopEO(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) {}
  void DhopDir(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dir, int disp) {}

  // dminus stuff /////////////////////////////////////////////////////////////

  void Dminus(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}
  void DminusDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {}

  // import/export ////////////////////////////////////////////////////////////

  void ImportPhysicalFermionSource(const FermionField& input, uint64_t input_n_virtual, FermionField& imported, uint64_t imported_n_virtual) {}
  void ImportUnphysicalFermion(const FermionField& input, uint64_t input_n_virtual, FermionField& imported, uint64_t imported_n_virtual) {}
  void ExportPhysicalFermionSolution(const FermionField& solution, uint64_t solution_n_virtual, FermionField& exported, uint64_t exported_n_virtual) {}
  void ExportPhysicalFermionSource(const FermionField& solution, uint64_t solution_n_virtual, FermionField& exported, uint64_t exported_n_virtual) {}

  /////////////////////////////////////////////////////////////////////////////
  //                      Member Functions (additional)                      //
  /////////////////////////////////////////////////////////////////////////////

public:

  // constructors /////////////////////////////////////////////////////////////

  MultiArgVirtualCoarsenedMatrix(const PVector<BasicLinkField>& Uc,
                                 const PVector<BasicLinkField>& UcSelfInv,
                                 GridCartesian&                 grid,
                                 GridRedBlackCartesian&         rbGrid,
                                 int                            makeHermitian)
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
    , stencil_(grid_, geom_.npoint, Even, geom_.directions, geom_.displacements, 0)
    , stencilEven_(cbGrid_, geom_.npoint, Even, geom_.directions, geom_.displacements, 0)
    , stencilOdd_(cbGrid_, geom_.npoint, Odd, geom_.directions, geom_.displacements, 0)
    , stencilMultiArg_(gridMultiArg_, geomMultiArg_.npoint, Even, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
    , stencilEvenMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint, Even, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
    , stencilOddMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint, Odd, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
    , Uc_(link_n_virtual_, grid_)
    , UcEven_(link_n_virtual_, cbGrid_)
    , UcOdd_(link_n_virtual_, cbGrid_)
    , UcSelfInv_(link_n_virtual_, grid_)
    , UcSelfInvEven_(link_n_virtual_, cbGrid_)
    , UcSelfInvOdd_(link_n_virtual_, cbGrid_)
    , tmpMultiArg_(gridMultiArg_)
  {
    grid_->show_decomposition();
    cbGrid_->show_decomposition();
    gridMultiArg_->show_decomposition();
    cbGridMultiArg_->show_decomposition();
    importGauge(Uc, UcSelfInv);
    pickCheckerboards();
    grid_msg("Constructed the latest coarse operator\n"); fflush(stdout);
    zeroCounters();
  }

  // helpers //////////////////////////////////////////////////////////////////

  void importGauge(const PVector<BasicLinkField>& Uc, const PVector<BasicLinkField>& UcSelfInv) {
    assert(Uc.size() == geom_.npoint * Uc_.size());
    assert(UcSelfInv.size() == UcSelfInv_.size());

    conformable(Grid(), Uc);
    conformable(Grid(), UcSelfInv);

    const int Nsimd        = SiteSpinor::Nsimd();
    const int Npoint       = geom_.npoint;
    const int NvirtualLink = link_n_virtual_;
    const int NvirtualFermion = fermion_n_virtual_;

    // NOTE: can't use PokeIndex here because of different tensor depths
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_member_v, Uc_member_p, AcceleratorWrite);
    VECTOR_VIEW_OPEN_POINTER(Uc,  Uc_arg_v,    Uc_arg_p,    AcceleratorRead);
    for(int v=0; v<NvirtualLink; v++) {
      const int v_row = v%NvirtualFermion; const int v_col = v/NvirtualFermion;
      for(int p=0; p<Npoint; p++) {
        accelerator_for(ss, Uc[0].Grid()->oSites(), Nsimd, { // NOTE: transpose here for better access pattern in application
          coalescedWrite(Uc_member_p[v_row*NvirtualFermion+v_col][ss](p), coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss]));
        });
      }
    }
    VECTOR_VIEW_CLOSE(Uc_member_v);
    VECTOR_VIEW_CLOSE(Uc_arg_v);

    for(int v=0; v<NvirtualLink; v++) UcSelfInv_[v] = UcSelfInv[v];
  }

  void pickCheckerboards() {
    // grid_warn("VirtualCoarsenedMatrix::pickCheckerboards still needs to be implemented\n");
    // TODO
  }

  void report() {
    RealD Nproc = grid_->_Nprocessors;
    RealD Nnode = grid_->NodeCount();
    RealD volume = 1;
    Coordinate latt = grid_->GlobalDimensions();
    for(int mu=0;mu<Nd;mu++) volume=volume*latt[mu];

    if ( MCalls > 0 ) {
      grid_msg("#### M calls report\n");
      grid_msg("CoarseOperator Number of Calls                         : %d\n", (int)MCalls);
      grid_msg("CoarseOperator MiscTime   /Calls, MiscTime    : %10.2f us, %10.2f us (= %6.2f %)\n", MMiscTime   /MCalls, MMiscTime,    MMiscTime   /MTotalTime*100);
      grid_msg("CoarseOperator ViewTime   /Calls, ViewTime    : %10.2f us, %10.2f us (= %6.2f %)\n", MViewTime   /MCalls, MViewTime,    MViewTime   /MTotalTime*100);
      grid_msg("CoarseOperator View2Time  /Calls, View2Time   : %10.2f us, %10.2f us (= %6.2f %)\n", MView2Time  /MCalls, MView2Time,   MView2Time  /MTotalTime*100);
      grid_msg("CoarseOperator CopyTime   /Calls, CopyTime    : %10.2f us, %10.2f us (= %6.2f %)\n", MCopyTime   /MCalls, MCopyTime,    MCopyTime   /MTotalTime*100);
      grid_msg("CoarseOperator CommTime   /Calls, CommTime    : %10.2f us, %10.2f us (= %6.2f %)\n", MCommTime   /MCalls, MCommTime,    MCommTime   /MTotalTime*100);
      grid_msg("CoarseOperator ComputeTime/Calls, ComputeTime : %10.2f us, %10.2f us (= %6.2f %)\n", MComputeTime/MCalls, MComputeTime, MComputeTime/MTotalTime*100);
      grid_msg("CoarseOperator TotalTime  /Calls, TotalTime   : %10.2f us, %10.2f us (= %6.2f %)\n", MTotalTime  /MCalls, MTotalTime,   MTotalTime  /MTotalTime*100);

      grid_msg("CoarseOperator Stencil\n"); stencil_.Report();
      grid_msg("CoarseOperator StencilEven\n"); stencilEven_.Report();
      grid_msg("CoarseOperator StencilOdd\n"); stencilOdd_.Report();
      grid_msg("CoarseOperator StencilMultiArg\n"); stencilMultiArg_.Report();

      // // Average the compute time
      // grid_->GlobalSum(MCommTime);
      // MCommTime/=Nproc;
      // RealD flop_per_site = 1320;
      // RealD mflops = flop_per_site*volume*MCalls/MCommTime/2; // 2 for red black counting
      // std::cout << GridLogMessage << "Average mflops/s per call                : " << mflops << std::endl;
      // std::cout << GridLogMessage << "Average mflops/s per call per rank       : " << mflops/Nproc << std::endl;
      // std::cout << GridLogMessage << "Average mflops/s per call per node       : " << mflops/Nnode << std::endl;

      // RealD Fullmflops = flop_per_site*volume*MCalls/(MTotalTime)/2; // 2 for red black counting
      // std::cout << GridLogMessage << "Average mflops/s per call (full)         : " << Fullmflops << std::endl;
      // std::cout << GridLogMessage << "Average mflops/s per call per rank (full): " << Fullmflops/Nproc << std::endl;
      // std::cout << GridLogMessage << "Average mflops/s per call per node (full): " << Fullmflops/Nnode << std::endl;
    }
  }

  // NOTE: this is only temporary -> TODO remove!
  ~MultiArgVirtualCoarsenedMatrix() { report(); grid_msg("\n"); fflush(stdout); }

  void zeroCounters() {
    MCalls       = 0; // ok
    MMiscTime    = 0;
    MViewTime    = 0;
    MView2Time   = 0;
    MCopyTime    = 0;
    MCommTime    = 0;
    MComputeTime = 0;
    MTotalTime   = 0;

    stencil_.ZeroCounters();
    stencilEven_.ZeroCounters();
    stencilOdd_.ZeroCounters();
  }

private: // some of these may be generally useful -> move somewhere else? TODO

  int numArg(const FermionField& a, uint64_t a_n_virtual, const FermionField& b, uint64_t b_n_virtual) const {
    int a_size = a.size(); int b_size = b.size();
    assert(a_size == b_size);
    assert(a_n_virtual == b_n_virtual);
    assert(a_n_virtual == fermion_n_virtual_);
    assert(a_size >= a_n_virtual);
    assert(a_size % a_n_virtual == 0);
    return a_size / a_n_virtual;
  }
};
