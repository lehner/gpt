/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Daniel Richtmann (daniel.richtmann@ur.de, https://github.com/lehner/gpt)
                  2020  Nils Meyer       (nils.meyer@ur.de,       https://github.com/lehner/gpt)
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
*/

// see Grid/qcd/action/fermion/WilsonCloverFermion for description
//
// Modifications done here:
//
// Grid: clover term = 12x12 matrix per site
//
// But: Only two diagonal 6x6 hermitian blocks are non-zero (also true in Grid, verified by running)
// Sufficient to store/transfer only the real parts of the diagonal and one triangular part
// 2 * (6 + 15 * 2) = 72 real or 36 complex words to be stored/transfered
//
// Here: Above but diagonal as complex numbers, i.e., need to store/transfer
// 2 * (6 * 2 + 15 * 2) = 84 real or 42 complex words
//
// Words per site and improvement compared to Grid (combined with the input and output spinors):
//
// - Grid:    2*12 + 12*12 = 168 words -> 1.00 x less
// - Minimal: 2*12 + 36    =  60 words -> 2.80 x less
// - Here:    2*12 + 42    =  66 words -> 2.55 x less
//
// These improvements directly translate to wall-clock time
//
// Data layout:
//
// - diagonal and triangle part as separate lattice fields,
//   this was faster than as 1 combined field on all tested machines
// - diagonal: as expected
// - triangle: store upper right triangle in row major order
// - graphical:
//        0  1  2  3  4
//           5  6  7  8
//              9 10 11 = upper right triangle indices
//                12 13
//                   14
//     0
//        1
//           2
//              3       = diagonal indices
//                 4
//                    5
//     0
//     1  5
//     2  6  9          = lower left triangle indices
//     3  7 10 12
//     4  8 11 13 14

template<class Impl>
class CompactWilsonCloverFermion : public WilsonFermion<Impl> {
  /////////////////////////////////////////////
  // Sizes
  /////////////////////////////////////////////

public:

  static constexpr int Nred      = Nc * Nhs;        // 6
  static constexpr int Nblock    = Nhs;             // 2
  static constexpr int Ndiagonal = Nred;            // 6
  static constexpr int Ntriangle = (Nred - 1) * Nc; // 15

  /////////////////////////////////////////////
  // Type definitions
  /////////////////////////////////////////////

public:

  static_assert(Nd == 4 && Nc == 3 && Ns == 4 && Impl::Dimension == 3, "Wrong dimensions");
  INHERIT_IMPL_TYPES(Impl);

  template<typename vtype> using iImplCloverDiagonal = iScalar<iVector<iVector<vtype, Ndiagonal>, Nblock>>;
  template<typename vtype> using iImplCloverTriangle = iScalar<iVector<iVector<vtype, Ntriangle>, Nblock>>;

  typedef iImplCloverDiagonal<Simd> SiteCloverDiagonal;
  typedef iImplCloverTriangle<Simd> SiteCloverTriangle;

  typedef Lattice<SiteCloverDiagonal> CloverDiagonalField;
  typedef Lattice<SiteCloverTriangle> CloverTriangleField;

  typedef WilsonCloverFermion<Impl> OriginalWilsonClover;
  typedef typename OriginalWilsonClover::SiteCloverType SiteCloverOriginal;
  typedef typename OriginalWilsonClover::CloverFieldType CloverOriginalField;

  typedef WilsonFermion<Impl> WilsonBase;

  typedef iSinglet<Simd>    SiteMask;
  typedef Lattice<SiteMask> MaskField;

  /////////////////////////////////////////////
  // Constructors
  /////////////////////////////////////////////

public:

  CompactWilsonCloverFermion(GaugeField& _Umu,
			     GridCartesian& Fgrid,
			     GridRedBlackCartesian& Hgrid,
			     const RealD _mass,
			     const RealD _csw_r = 0.0,
			     const RealD _csw_t = 0.0,
			     const RealD _cF = 1.0,
			     const WilsonAnisotropyCoefficients& clover_anisotropy = WilsonAnisotropyCoefficients(),
			     const ImplParams& impl_p = ImplParams())
    : WilsonBase(_Umu, Fgrid, Hgrid, _mass, impl_p, clover_anisotropy)
    , csw_r(_csw_r)
    , csw_t(_csw_t)
    , cF(_cF)
    , open_boundaries(impl_p.boundary_phases[Nd-1] == 0.0)
    , Diagonal(&Fgrid),        Triangle(&Fgrid)
    , DiagonalEven(&Hgrid),    TriangleEven(&Hgrid)
    , DiagonalOdd(&Hgrid),     TriangleOdd(&Hgrid)
    , DiagonalInv(&Fgrid),     TriangleInv(&Fgrid)
    , DiagonalInvEven(&Hgrid), TriangleInvEven(&Hgrid)
    , DiagonalInvOdd(&Hgrid),  TriangleInvOdd(&Hgrid)
    , Tmp(&Fgrid)
    , BoundaryMask(&Fgrid)
    , BoundaryMaskEven(&Hgrid), BoundaryMaskOdd(&Hgrid)
  {
    csw_r *= 0.5;
    csw_t *= 0.5;
    if (clover_anisotropy.isAnisotropic)
      csw_r /= clover_anisotropy.xi_0;

    ImportGauge(_Umu);
    if (open_boundaries) SetupMasks();
  }

  /////////////////////////////////////////////
  // Member functions (implementing interface)
  /////////////////////////////////////////////

public:

  virtual void Instantiatable() {};
  int          ConstEE()     override { return 0; };
  int          isTrivialEE() override { return 0; };


  void Dhop(const FermionField& in, FermionField& out, int dag) override {
    WilsonBase::Dhop(in, out, dag);
    if(open_boundaries) ApplyBoundaryMask(out);
  }


  void DhopOE(const FermionField& in, FermionField& out, int dag) override {
    WilsonBase::DhopOE(in, out, dag);
    if(open_boundaries) ApplyBoundaryMask(out);
  }


  void DhopEO(const FermionField& in, FermionField& out, int dag) override {
    WilsonBase::DhopEO(in, out, dag);
    if(open_boundaries) ApplyBoundaryMask(out);
  }


  void M(const FermionField& in, FermionField& out) override {
    out.Checkerboard() = in.Checkerboard();
    WilsonBase::Dhop(in, out, DaggerNo); // call base to save applying bc
    Mooee(in, Tmp);
    axpy(out, 1.0, out, Tmp);
    if(open_boundaries) ApplyBoundaryMask(out);
  }


  void Mdag(const FermionField& in, FermionField& out) override {
    out.Checkerboard() = in.Checkerboard();
    WilsonBase::Dhop(in, out, DaggerYes);  // call base to save applying bc
    MooeeDag(in, Tmp);
    axpy(out, 1.0, out, Tmp);
    if(open_boundaries) ApplyBoundaryMask(out);
  }


  void Meooe(const FermionField& in, FermionField& out) override {
    WilsonBase::Meooe(in, out);
    if(open_boundaries) ApplyBoundaryMask(out);
  }


  void MeooeDag(const FermionField& in, FermionField& out) override {
    WilsonBase::MeooeDag(in, out);
    if(open_boundaries) ApplyBoundaryMask(out);
  }


  void Mooee(const FermionField& in, FermionField& out) override {
    if(in.Grid()->_isCheckerBoarded) {
      if(in.Checkerboard() == Odd) {
        MooeeInternal(in, out, DiagonalOdd, TriangleOdd);
      } else {
        MooeeInternal(in, out, DiagonalEven, TriangleEven);
      }
    } else {
      MooeeInternal(in, out, Diagonal, Triangle);
    }
    if(open_boundaries) ApplyBoundaryMask(out);
  }


  void MooeeDag(const FermionField& in, FermionField& out) override {
    Mooee(in, out); // blocks are hermitian
  }


  void MooeeInv(const FermionField& in, FermionField& out) override {
    if(in.Grid()->_isCheckerBoarded) {
      if(in.Checkerboard() == Odd) {
        MooeeInternal(in, out, DiagonalInvOdd, TriangleInvOdd);
      } else {
        MooeeInternal(in, out, DiagonalInvEven, TriangleInvEven);
      }
    } else {
      MooeeInternal(in, out, DiagonalInv, TriangleInv);
    }
    if(open_boundaries) ApplyBoundaryMask(out);
  }


  void MooeeInvDag(const FermionField& in, FermionField& out) override {
    MooeeInv(in, out); // blocks are hermitian
  }


  void Mdir(const FermionField& in, FermionField& out, int dir, int disp) override {
    WilsonBase::Mdir(in, out, dir, disp);
    if(open_boundaries) ApplyBoundaryMask(out);
  }


  void MDeriv(GaugeField& force, const FermionField& X, const FermionField& Y, int dag) override {
    assert(!open_boundaries); // TODO check for open bc

    // NOTE: code copied from original clover term
    conformable(X.Grid(), Y.Grid());
    conformable(X.Grid(), force.Grid());
    GaugeLinkField force_mu(force.Grid()), lambda(force.Grid());
    GaugeField clover_force(force.Grid());
    PropagatorField Lambda(force.Grid());

    // Guido: Here we are hitting some performance issues:
    // need to extract the components of the DoubledGaugeField
    // for each call
    // Possible solution
    // Create a vector object to store them? (cons: wasting space)
    std::vector<GaugeLinkField> U(Nd, this->Umu.Grid());

    Impl::extractLinkField(U, this->Umu);

    force = Zero();
    // Derivative of the Wilson hopping term
    this->DhopDeriv(force, X, Y, dag);

    ///////////////////////////////////////////////////////////
    // Clover term derivative
    ///////////////////////////////////////////////////////////
    Impl::outerProductImpl(Lambda, X, Y);
    //std::cout << "Lambda:" << Lambda << std::endl;

    Gamma::Algebra sigma[] = {
        Gamma::Algebra::SigmaXY,
        Gamma::Algebra::SigmaXZ,
        Gamma::Algebra::SigmaXT,
        Gamma::Algebra::MinusSigmaXY,
        Gamma::Algebra::SigmaYZ,
        Gamma::Algebra::SigmaYT,
        Gamma::Algebra::MinusSigmaXZ,
        Gamma::Algebra::MinusSigmaYZ,
        Gamma::Algebra::SigmaZT,
        Gamma::Algebra::MinusSigmaXT,
        Gamma::Algebra::MinusSigmaYT,
        Gamma::Algebra::MinusSigmaZT};

    /*
      sigma_{\mu \nu}=
      | 0         sigma[0]  sigma[1]  sigma[2] |
      | sigma[3]    0       sigma[4]  sigma[5] |
      | sigma[6]  sigma[7]     0      sigma[8] |
      | sigma[9]  sigma[10] sigma[11]   0      |
    */

    int count = 0;
    clover_force = Zero();
    for (int mu = 0; mu < 4; mu++)
    {
      force_mu = Zero();
      for (int nu = 0; nu < 4; nu++)
      {
        if (mu == nu)
        continue;

        RealD factor;
        if (nu == 4 || mu == 4)
        {
          factor = 2.0 * csw_t;
        }
        else
        {
          factor = 2.0 * csw_r;
        }
        PropagatorField Slambda = Gamma(sigma[count]) * Lambda; // sigma checked
        Impl::TraceSpinImpl(lambda, Slambda);                   // traceSpin ok
        force_mu -= factor*Cmunu(U, lambda, mu, nu);                   // checked
        count++;
      }

      pokeLorentz(clover_force, U[mu] * force_mu, mu);
    }
    //clover_force *= csw;
    force += clover_force;
  }


  void MooDeriv(GaugeField& mat, const FermionField& U, const FermionField& V, int dag) override {
    assert(0);
  }


  void MeeDeriv(GaugeField& mat, const FermionField& U, const FermionField& V, int dag) override {
    assert(0);
  }

  /////////////////////////////////////////////
  // Member functions (internals/kernels)
  /////////////////////////////////////////////

  void MooeeInternal(const FermionField&        in,
                     FermionField&              out,
                     const CloverDiagonalField& diagonal,
                     const CloverTriangleField& triangle) {
    assert(in.Checkerboard() == Odd || in.Checkerboard() == Even);
    out.Checkerboard() = in.Checkerboard();
    conformable(in, out);
    conformable(in, diagonal);
    conformable(in, triangle);

#if defined(GRID_CUDA) || defined(GRID_HIP)
    MooeeKernel_gpu(in.oSites(), in, out, diagonal, triangle);
#else
    MooeeKernel_cpu(in.oSites(), in, out, diagonal, triangle);
#endif
  }


  void MooeeKernel_gpu(int                        Nsite,
                       const FermionField&        in,
                       FermionField&              out,
                       const CloverDiagonalField& diagonal,
                       const CloverTriangleField& triangle) {
    autoView(diagonal_v, diagonal, AcceleratorRead);
    autoView(triangle_v, triangle, AcceleratorRead);
    autoView(in_v,       in,       AcceleratorRead);
    autoView(out_v,      out,      AcceleratorWrite);

    typedef decltype(coalescedRead(out_v[0])) CalcSpinor;

    accelerator_for(ss, Nsite, Simd::Nsimd(), {
      CalcSpinor res;
      CalcSpinor in_t = in_v(ss);
      auto diagonal_t = diagonal_v(ss);
      auto triangle_t = triangle_v(ss);
      for(int block=0; block<Nhs; block++) {
        int s_start = block*Nhs;
        for(int i=0; i<Nred; i++) {
          int si = s_start + i/Nc, ci = i%Nc;
          res()(si)(ci) = diagonal_t()(block)(i) * in_t()(si)(ci);
          for(int j=0; j<Nred; j++) {
            if (j == i) continue;
            int sj = s_start + j/Nc, cj = j%Nc;
            res()(si)(ci) = res()(si)(ci) + triangle_elem(triangle_t, block, i, j) * in_t()(sj)(cj);
          };
        };
      };
      coalescedWrite(out_v[ss], res);
    });
  }


  void MooeeKernel_cpu(int                        Nsite,
                       const FermionField&        in,
                       FermionField&              out,
                       const CloverDiagonalField& diagonal,
                       const CloverTriangleField& triangle) {
    autoView(diagonal_v, diagonal, CpuRead);
    autoView(triangle_v, triangle, CpuRead);
    autoView(in_v,       in,       CpuRead);
    autoView(out_v,      out,      CpuWrite);

    typedef SiteSpinor CalcSpinor;

#if defined(A64FX) || defined(A64FXFIXEDSIZE)
#define PREFETCH_CLOVER(BASE) {                                     \
    uint64_t base;                                                  \
    int pf_dist_L1 = 1;                                             \
    int pf_dist_L2 = -5; /* -> penalty -> disable */                \
                                                                    \
    if ((pf_dist_L1 >= 0) && (ss + pf_dist_L1 < Nsite)) {           \
      base = (uint64_t)&diag_t()(pf_dist_L1+BASE)(0);               \
      svprfd(svptrue_b64(), (int64_t*)(base +    0), SV_PLDL1STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base +  256), SV_PLDL1STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base +  512), SV_PLDL1STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base +  768), SV_PLDL1STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base + 1024), SV_PLDL1STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base + 1280), SV_PLDL1STRM); \
    }                                                               \
                                                                    \
    if ((pf_dist_L2 >= 0) && (ss + pf_dist_L2 < Nsite)) {           \
      base = (uint64_t)&diag_t()(pf_dist_L2+BASE)(0);               \
      svprfd(svptrue_b64(), (int64_t*)(base +    0), SV_PLDL2STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base +  256), SV_PLDL2STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base +  512), SV_PLDL2STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base +  768), SV_PLDL2STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base + 1024), SV_PLDL2STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base + 1280), SV_PLDL2STRM); \
    }                                                               \
  }
// TODO: Implement/generalize this for other architectures
// I played around a bit on KNL (see below) but didn't bring anything
// #elif defined(AVX512)
// #define PREFETCH_CLOVER(BASE) {                              \
//     uint64_t base;                                           \
//     int pf_dist_L1 = 1;                                      \
//     int pf_dist_L2 = +4;                                     \
//                                                              \
//     if ((pf_dist_L1 >= 0) && (ss + pf_dist_L1 < Nsite)) {    \
//       base = (uint64_t)&diag_t()(pf_dist_L1+BASE)(0);        \
//       _mm_prefetch((const char*)(base +    0), _MM_HINT_T0); \
//       _mm_prefetch((const char*)(base +   64), _MM_HINT_T0); \
//       _mm_prefetch((const char*)(base +  128), _MM_HINT_T0); \
//       _mm_prefetch((const char*)(base +  192), _MM_HINT_T0); \
//       _mm_prefetch((const char*)(base +  256), _MM_HINT_T0); \
//       _mm_prefetch((const char*)(base +  320), _MM_HINT_T0); \
//     }                                                        \
//                                                              \
//     if ((pf_dist_L2 >= 0) && (ss + pf_dist_L2 < Nsite)) {    \
//       base = (uint64_t)&diag_t()(pf_dist_L2+BASE)(0);        \
//       _mm_prefetch((const char*)(base +    0), _MM_HINT_T1); \
//       _mm_prefetch((const char*)(base +   64), _MM_HINT_T1); \
//       _mm_prefetch((const char*)(base +  128), _MM_HINT_T1); \
//       _mm_prefetch((const char*)(base +  192), _MM_HINT_T1); \
//       _mm_prefetch((const char*)(base +  256), _MM_HINT_T1); \
//       _mm_prefetch((const char*)(base +  320), _MM_HINT_T1); \
//     }                                                        \
//   }
#else
#define PREFETCH_CLOVER(BASE)
#endif

    thread_for(ss, Nsite, {
      CalcSpinor res;
      CalcSpinor in_t = in_v[ss];
      auto diag_t     = diagonal_v[ss]; // "diag" instead of "diagonal" here to make code below easier to read
      auto triangle_t = triangle_v[ss];

      // upper half
      PREFETCH_CLOVER(0);

      auto in_cc_0_0 = conjugate(in_t()(0)(0)); // Nils: reduces number
      auto in_cc_0_1 = conjugate(in_t()(0)(1)); // of conjugates from
      auto in_cc_0_2 = conjugate(in_t()(0)(2)); // 30 to 20
      auto in_cc_1_0 = conjugate(in_t()(1)(0));
      auto in_cc_1_1 = conjugate(in_t()(1)(1));

      res()(0)(0) =               diag_t()(0)( 0) * in_t()(0)(0)
                  +           triangle_t()(0)( 0) * in_t()(0)(1)
                  +           triangle_t()(0)( 1) * in_t()(0)(2)
                  +           triangle_t()(0)( 2) * in_t()(1)(0)
                  +           triangle_t()(0)( 3) * in_t()(1)(1)
                  +           triangle_t()(0)( 4) * in_t()(1)(2);

      res()(0)(1) =           triangle_t()(0)( 0) * in_cc_0_0;
      res()(0)(1) =               diag_t()(0)( 1) * in_t()(0)(1)
                  +           triangle_t()(0)( 5) * in_t()(0)(2)
                  +           triangle_t()(0)( 6) * in_t()(1)(0)
                  +           triangle_t()(0)( 7) * in_t()(1)(1)
                  +           triangle_t()(0)( 8) * in_t()(1)(2)
                  + conjugate(       res()(0)( 1));

      res()(0)(2) =           triangle_t()(0)( 1) * in_cc_0_0
                  +           triangle_t()(0)( 5) * in_cc_0_1;
      res()(0)(2) =               diag_t()(0)( 2) * in_t()(0)(2)
                  +           triangle_t()(0)( 9) * in_t()(1)(0)
                  +           triangle_t()(0)(10) * in_t()(1)(1)
                  +           triangle_t()(0)(11) * in_t()(1)(2)
                  + conjugate(       res()(0)( 2));

      res()(1)(0) =           triangle_t()(0)( 2) * in_cc_0_0
                  +           triangle_t()(0)( 6) * in_cc_0_1
                  +           triangle_t()(0)( 9) * in_cc_0_2;
      res()(1)(0) =               diag_t()(0)( 3) * in_t()(1)(0)
                  +           triangle_t()(0)(12) * in_t()(1)(1)
                  +           triangle_t()(0)(13) * in_t()(1)(2)
                  + conjugate(       res()(1)( 0));

      res()(1)(1) =           triangle_t()(0)( 3) * in_cc_0_0
                  +           triangle_t()(0)( 7) * in_cc_0_1
                  +           triangle_t()(0)(10) * in_cc_0_2
                  +           triangle_t()(0)(12) * in_cc_1_0;
      res()(1)(1) =               diag_t()(0)( 4) * in_t()(1)(1)
                  +           triangle_t()(0)(14) * in_t()(1)(2)
                  + conjugate(       res()(1)( 1));

      res()(1)(2) =           triangle_t()(0)( 4) * in_cc_0_0
                  +           triangle_t()(0)( 8) * in_cc_0_1
                  +           triangle_t()(0)(11) * in_cc_0_2
                  +           triangle_t()(0)(13) * in_cc_1_0
                  +           triangle_t()(0)(14) * in_cc_1_1;
      res()(1)(2) =               diag_t()(0)( 5) * in_t()(1)(2)
                  + conjugate(       res()(1)( 2));

      vstream(out_v[ss]()(0)(0), res()(0)(0));
      vstream(out_v[ss]()(0)(1), res()(0)(1));
      vstream(out_v[ss]()(0)(2), res()(0)(2));
      vstream(out_v[ss]()(1)(0), res()(1)(0));
      vstream(out_v[ss]()(1)(1), res()(1)(1));
      vstream(out_v[ss]()(1)(2), res()(1)(2));

      // lower half
      PREFETCH_CLOVER(1);

      auto in_cc_2_0 = conjugate(in_t()(2)(0));
      auto in_cc_2_1 = conjugate(in_t()(2)(1));
      auto in_cc_2_2 = conjugate(in_t()(2)(2));
      auto in_cc_3_0 = conjugate(in_t()(3)(0));
      auto in_cc_3_1 = conjugate(in_t()(3)(1));

      res()(2)(0) =               diag_t()(1)( 0) * in_t()(2)(0)
                  +           triangle_t()(1)( 0) * in_t()(2)(1)
                  +           triangle_t()(1)( 1) * in_t()(2)(2)
                  +           triangle_t()(1)( 2) * in_t()(3)(0)
                  +           triangle_t()(1)( 3) * in_t()(3)(1)
                  +           triangle_t()(1)( 4) * in_t()(3)(2);

      res()(2)(1) =           triangle_t()(1)( 0) * in_cc_2_0;
      res()(2)(1) =               diag_t()(1)( 1) * in_t()(2)(1)
                  +           triangle_t()(1)( 5) * in_t()(2)(2)
                  +           triangle_t()(1)( 6) * in_t()(3)(0)
                  +           triangle_t()(1)( 7) * in_t()(3)(1)
                  +           triangle_t()(1)( 8) * in_t()(3)(2)
                  + conjugate(       res()(2)( 1));

      res()(2)(2) =           triangle_t()(1)( 1) * in_cc_2_0
                  +           triangle_t()(1)( 5) * in_cc_2_1;
      res()(2)(2) =               diag_t()(1)( 2) * in_t()(2)(2)
                  +           triangle_t()(1)( 9) * in_t()(3)(0)
                  +           triangle_t()(1)(10) * in_t()(3)(1)
                  +           triangle_t()(1)(11) * in_t()(3)(2)
                  + conjugate(       res()(2)( 2));

      res()(3)(0) =           triangle_t()(1)( 2) * in_cc_2_0
                  +           triangle_t()(1)( 6) * in_cc_2_1
                  +           triangle_t()(1)( 9) * in_cc_2_2;
      res()(3)(0) =               diag_t()(1)( 3) * in_t()(3)(0)
                  +           triangle_t()(1)(12) * in_t()(3)(1)
                  +           triangle_t()(1)(13) * in_t()(3)(2)
                  + conjugate(       res()(3)( 0));

      res()(3)(1) =           triangle_t()(1)( 3) * in_cc_2_0
                  +           triangle_t()(1)( 7) * in_cc_2_1
                  +           triangle_t()(1)(10) * in_cc_2_2
                  +           triangle_t()(1)(12) * in_cc_3_0;
      res()(3)(1) =               diag_t()(1)( 4) * in_t()(3)(1)
                  +           triangle_t()(1)(14) * in_t()(3)(2)
                  + conjugate(       res()(3)( 1));

      res()(3)(2) =           triangle_t()(1)( 4) * in_cc_2_0
                  +           triangle_t()(1)( 8) * in_cc_2_1
                  +           triangle_t()(1)(11) * in_cc_2_2
                  +           triangle_t()(1)(13) * in_cc_3_0
                  +           triangle_t()(1)(14) * in_cc_3_1;
      res()(3)(2) =               diag_t()(1)( 5) * in_t()(3)(2)
                  + conjugate(       res()(3)( 2));

      vstream(out_v[ss]()(2)(0), res()(2)(0));
      vstream(out_v[ss]()(2)(1), res()(2)(1));
      vstream(out_v[ss]()(2)(2), res()(2)(2));
      vstream(out_v[ss]()(3)(0), res()(3)(0));
      vstream(out_v[ss]()(3)(1), res()(3)(1));
      vstream(out_v[ss]()(3)(2), res()(3)(2));
    });
  }

  /////////////////////////////////////////////
  // Helpers
  /////////////////////////////////////////////

  void ImportGauge(const GaugeField& _Umu) override {
    // NOTE: parts copied from original implementation

    // Import gauge into base class
    double t0 = usecond();
    WilsonBase::ImportGauge(_Umu); // NOTE: called here and in wilson constructor -> performed twice, but can't avoid that

    // Initialize temporary variables
    double t1 = usecond();
    conformable(_Umu.Grid(), this->GaugeGrid());
    GridBase* grid = _Umu.Grid();
    typename Impl::GaugeLinkField Bx(grid), By(grid), Bz(grid), Ex(grid), Ey(grid), Ez(grid);
    CloverOriginalField TmpOriginal(grid);

    // Compute the field strength terms mu>nu
    double t2 = usecond();
    WilsonLoops<Impl>::FieldStrength(Bx, _Umu, Zdir, Ydir);
    WilsonLoops<Impl>::FieldStrength(By, _Umu, Zdir, Xdir);
    WilsonLoops<Impl>::FieldStrength(Bz, _Umu, Ydir, Xdir);
    WilsonLoops<Impl>::FieldStrength(Ex, _Umu, Tdir, Xdir);
    WilsonLoops<Impl>::FieldStrength(Ey, _Umu, Tdir, Ydir);
    WilsonLoops<Impl>::FieldStrength(Ez, _Umu, Tdir, Zdir);

    // Compute the Clover Operator acting on Colour and Spin
    // multiply here by the clover coefficients for the anisotropy
    double t3 = usecond();
    TmpOriginal  = fillCloverYZ(Bx) * csw_r;
    TmpOriginal += fillCloverXZ(By) * csw_r;
    TmpOriginal += fillCloverXY(Bz) * csw_r;
    TmpOriginal += fillCloverXT(Ex) * csw_t;
    TmpOriginal += fillCloverYT(Ey) * csw_t;
    TmpOriginal += fillCloverZT(Ez) * csw_t;
    TmpOriginal += this->diag_mass;

    // Convert the data layout of the clover term
    double t4 = usecond();
    ConvertLayout(TmpOriginal, Diagonal, Triangle);

    // Possible modify the boundary values
    double t5 = usecond();
    if(open_boundaries) ModifyBoundaries(Diagonal, Triangle);

    // Invert the clover term in the improved layout
    double t6 = usecond();
    Invert(Diagonal, Triangle, DiagonalInv, TriangleInv);

    // Fill the remaining clover fields
    double t7 = usecond();
    pickCheckerboard(Even, DiagonalEven,    Diagonal);
    pickCheckerboard(Even, TriangleEven,    Triangle);
    pickCheckerboard(Odd,  DiagonalOdd,     Diagonal);
    pickCheckerboard(Odd,  TriangleOdd,     Triangle);
    pickCheckerboard(Even, DiagonalInvEven, DiagonalInv);
    pickCheckerboard(Even, TriangleInvEven, TriangleInv);
    pickCheckerboard(Odd,  DiagonalInvOdd,  DiagonalInv);
    pickCheckerboard(Odd,  TriangleInvOdd,  TriangleInv);

    // Report timings
    double t8 = usecond();
#if 0
    std::cout << GridLogMessage << "CompactWilsonCloverFermion::ImportGauge timings:"
              << " WilsonFermion::Importgauge = " << (t1 - t0) / 1e6
              << ", allocations = "               << (t2 - t1) / 1e6
              << ", field strength = "            << (t3 - t2) / 1e6
              << ", fill clover = "               << (t4 - t3) / 1e6
              << ", convert = "                   << (t5 - t4) / 1e6
              << ", boundaries = "                << (t6 - t5) / 1e6
              << ", inversions = "                << (t7 - t6) / 1e6
              << ", pick cbs = "                  << (t8 - t7) / 1e6
              << ", total = "                     << (t8 - t0) / 1e6
              << std::endl;
#endif
  }


  void SetupMasks() {
    GridBase* grid = BoundaryMask.Grid();
    int t_dir = Nd-1;
    Lattice<iScalar<vInteger>> t_coor(grid);
    LatticeCoordinate(t_coor, t_dir);
    int T = grid->GlobalDimensions()[t_dir];

    decltype(BoundaryMask) zeroMask(grid); zeroMask = Zero();
    BoundaryMask = 1.0;
    BoundaryMask = where(t_coor == 0,   zeroMask, BoundaryMask);
    BoundaryMask = where(t_coor == T-1, zeroMask, BoundaryMask);

    pickCheckerboard(Even, BoundaryMaskEven, BoundaryMask);
    pickCheckerboard(Odd,  BoundaryMaskOdd,  BoundaryMask);
  }


  // TODO: move this to matrix.h?
  void Invert(const CloverDiagonalField& diagonal,
              const CloverTriangleField& triangle,
              CloverDiagonalField&       diagonalInv,
              CloverTriangleField&       triangleInv) {
    conformable(diagonal, diagonalInv);
    conformable(triangle, triangleInv);
    conformable(diagonal, triangle);

    diagonalInv.Checkerboard() = diagonal.Checkerboard();
    triangleInv.Checkerboard() = triangle.Checkerboard();

    GridBase* grid = diagonal.Grid();

    long lsites = grid->lSites();

    typedef typename SiteCloverDiagonal::scalar_object scalar_object_diagonal;
    typedef typename SiteCloverTriangle::scalar_object scalar_object_triangle;

    autoView(diagonal_v,  diagonal,  CpuRead);
    autoView(triangle_v,  triangle,  CpuRead);
    autoView(diagonalInv_v, diagonalInv, CpuWrite);
    autoView(triangleInv_v, triangleInv, CpuWrite);

    thread_for(site, lsites, { // NOTE: Not on GPU because of Eigen & (peek/poke)LocalSite
      Eigen::MatrixXcd clover_inv_eigen = Eigen::MatrixXcd::Zero(Ns*Nc, Ns*Nc);
      Eigen::MatrixXcd clover_eigen = Eigen::MatrixXcd::Zero(Ns*Nc, Ns*Nc);

      scalar_object_diagonal diagonal_tmp     = Zero();
      scalar_object_diagonal diagonal_inv_tmp = Zero();
      scalar_object_triangle triangle_tmp     = Zero();
      scalar_object_triangle triangle_inv_tmp = Zero();

      Coordinate lcoor;
      grid->LocalIndexToLocalCoor(site, lcoor);

      peekLocalSite(diagonal_tmp, diagonal_v, lcoor);
      peekLocalSite(triangle_tmp, triangle_v, lcoor);

      // TODO: can we save time here by inverting the two 6x6 hermitian matrices separately?
      for (long s_row=0;s_row<Ns;s_row++) {
        for (long s_col=0;s_col<Ns;s_col++) {
          if(abs(s_row - s_col) > 1 || s_row + s_col == 3) continue;
          int block       = s_row / Nhs;
          int s_row_block = s_row % Nhs;
          int s_col_block = s_col % Nhs;
          for (long c_row=0;c_row<Nc;c_row++) {
            for (long c_col=0;c_col<Nc;c_col++) {
              int i = s_row_block * Nc + c_row;
              int j = s_col_block * Nc + c_col;
              if(i == j)
                clover_eigen(s_row*Nc+c_row, s_col*Nc+c_col) = static_cast<ComplexD>(TensorRemove(diagonal_tmp()(block)(i)));
              else
                clover_eigen(s_row*Nc+c_row, s_col*Nc+c_col) = static_cast<ComplexD>(TensorRemove(triangle_elem(triangle_tmp, block, i, j)));
            }
          }
        }
      }

      clover_inv_eigen = clover_eigen.inverse();

      for (long s_row=0;s_row<Ns;s_row++) {
        for (long s_col=0;s_col<Ns;s_col++) {
          if(abs(s_row - s_col) > 1 || s_row + s_col == 3) continue;
          int block       = s_row / Nhs;
          int s_row_block = s_row % Nhs;
          int s_col_block = s_col % Nhs;
          for (long c_row=0;c_row<Nc;c_row++) {
            for (long c_col=0;c_col<Nc;c_col++) {
              int i = s_row_block * Nc + c_row;
              int j = s_col_block * Nc + c_col;
              if(i == j)
                diagonal_inv_tmp()(block)(i) = clover_inv_eigen(s_row*Nc+c_row, s_col*Nc+c_col);
              else if(i < j)
                triangle_inv_tmp()(block)(triangle_index(i, j)) = clover_inv_eigen(s_row*Nc+c_row, s_col*Nc+c_col);
              else
                continue;
            }
          }
        }
      }

      pokeLocalSite(diagonal_inv_tmp, diagonalInv_v, lcoor);
      pokeLocalSite(triangle_inv_tmp, triangleInv_v, lcoor);
    });
  }


  void ConvertLayout(const CloverOriginalField& full,
                     CloverDiagonalField&       diagonal,
                     CloverTriangleField&       triangle) {
    conformable(full, diagonal);
    conformable(full, triangle);

    diagonal.Checkerboard() = full.Checkerboard();
    triangle.Checkerboard() = full.Checkerboard();

    autoView(full_v,     full,     AcceleratorRead);
    autoView(diagonal_v, diagonal, AcceleratorWrite);
    autoView(triangle_v, triangle, AcceleratorWrite);

    // NOTE: this function cannot be 'private' since nvcc forbids this for kernels
    accelerator_for(ss, full.Grid()->oSites(), 1, {
      for(int s_row = 0; s_row < Ns; s_row++) {
        for(int s_col = 0; s_col < Ns; s_col++) {
          if(abs(s_row - s_col) > 1 || s_row + s_col == 3) continue;
          int block       = s_row / Nhs;
          int s_row_block = s_row % Nhs;
          int s_col_block = s_col % Nhs;
          for(int c_row = 0; c_row < Nc; c_row++) {
            for(int c_col = 0; c_col < Nc; c_col++) {
              int i = s_row_block * Nc + c_row;
              int j = s_col_block * Nc + c_col;
              if(i == j)
                diagonal_v[ss]()(block)(i) = full_v[ss]()(s_row, s_col)(c_row, c_col);
              else if(i < j)
                triangle_v[ss]()(block)(triangle_index(i, j)) = full_v[ss]()(s_row, s_col)(c_row, c_col);
              else
                continue;
            }
          }
        }
      }
    });
  }


  void ConvertLayout(const CloverDiagonalField& diagonal,
                     const CloverTriangleField& triangle,
                     CloverOriginalField&       full) {
    conformable(full, diagonal);
    conformable(full, triangle);

    full.Checkerboard() = diagonal.Checkerboard();

    full = Zero();

    autoView(diagonal_v, diagonal, AcceleratorRead);
    autoView(triangle_v, triangle, AcceleratorRead);
    autoView(full_v,     full,     AcceleratorWrite);

    // NOTE: this function cannot be 'private' since nvcc forbids this for kernels
    accelerator_for(ss, full.Grid()->oSites(), 1, {
      for(int s_row = 0; s_row < Ns; s_row++) {
        for(int s_col = 0; s_col < Ns; s_col++) {
          if(abs(s_row - s_col) > 1 || s_row + s_col == 3) continue;
          int block       = s_row / Nhs;
          int s_row_block = s_row % Nhs;
          int s_col_block = s_col % Nhs;
          for(int c_row = 0; c_row < Nc; c_row++) {
            for(int c_col = 0; c_col < Nc; c_col++) {
              int i = s_row_block * Nc + c_row;
              int j = s_col_block * Nc + c_col;
              if(i == j)
                full_v[ss]()(s_row, s_col)(c_row, c_col) = diagonal_v[ss]()(block)(i);
              else
                full_v[ss]()(s_row, s_col)(c_row, c_col) = triangle_elem(triangle_v[ss], block, i, j);
            }
          }
        }
      }
    });
  }


  void ModifyBoundaries(CloverDiagonalField& diagonal, CloverTriangleField& triangle) const {
    // Checks/grid
    double t0 = usecond();
    conformable(diagonal, triangle);
    GridBase* grid = diagonal.Grid();

    // Determine the boundary coordinates/sites
    double t1 = usecond();
    int t_dir = Nd - 1;
    Lattice<iScalar<vInteger>> t_coor(grid);
    LatticeCoordinate(t_coor, t_dir);
    int T = grid->GlobalDimensions()[t_dir];

    // Set off-diagonal parts at boundary to zero -- OK
    double t2 = usecond();
    CloverTriangleField zeroTriangle(grid);
    zeroTriangle.Checkerboard() = triangle.Checkerboard();
    zeroTriangle = Zero();
    triangle = where(t_coor == 0,   zeroTriangle, triangle);
    triangle = where(t_coor == T-1, zeroTriangle, triangle);

    // Set diagonal to unity (scaled correctly) -- OK
    double t3 = usecond();
    CloverDiagonalField tmp(grid);
    tmp.Checkerboard() = diagonal.Checkerboard();
    tmp                = -1.0 * csw_t + this->diag_mass;
    diagonal           = where(t_coor == 0,   tmp, diagonal);
    diagonal           = where(t_coor == T-1, tmp, diagonal);

    // Correct values next to boundary
    double t4 = usecond();
    if(cF != 1.0) {
      tmp = cF - 1.0;
      tmp += diagonal;
      diagonal = where(t_coor == 1,   tmp, diagonal);
      diagonal = where(t_coor == T-2, tmp, diagonal);
    }

    // Report timings
    double t5 = usecond();
#if 0
    std::cout << GridLogMessage << "CompactWilsonCloverFermion::ModifyBoundaries timings:"
              << " checks = "          << (t1 - t0) / 1e6
              << ", coordinate = "     << (t2 - t1) / 1e6
              << ", off-diag zero = "  << (t3 - t2) / 1e6
              << ", diagonal unity = " << (t4 - t3) / 1e6
              << ", near-boundary = "  << (t5 - t4) / 1e6
              << ", total = "          << (t5 - t0) / 1e6
              << std::endl;
#endif
   }


  template<class Field>
  strong_inline void ApplyBoundaryMask(Field& f) const {
    if(f.Grid()->_isCheckerBoarded) {
      if(f.Checkerboard() == Odd) {
        ApplyBoundaryMask(f, BoundaryMaskOdd);
      } else {
        ApplyBoundaryMask(f, BoundaryMaskEven);
      }
    } else {
      ApplyBoundaryMask(f, BoundaryMask);
    }
  }
  template<class Field, class Mask>
  strong_inline void ApplyBoundaryMask(Field& f, const Mask& m) const {
    conformable(f, m);
    auto grid  = f.Grid();
    const int Nsite = grid->oSites();
    const int Nsimd = grid->Nsimd();
    autoView(f_v, f, AcceleratorWrite);
    autoView(m_v, m, AcceleratorRead);
    // NOTE: this function cannot be 'private' since nvcc forbids this for kernels
    accelerator_for(ss, Nsite, Nsimd, {
      coalescedWrite(f_v[ss], m_v(ss) * f_v(ss));
    });
  }


  CloverOriginalField fillCloverYZ(const GaugeLinkField& F) {
    // NOTE: code copied from original clover term
    CloverOriginalField T(F.Grid());
    T = Zero();

    autoView(T_v, T, AcceleratorWrite);
    autoView(F_v, F, AcceleratorRead);
    accelerator_for(i, T.Grid()->oSites(), 1, {
      T_v[i]()(0, 1) = timesMinusI(F_v[i]()());
      T_v[i]()(1, 0) = timesMinusI(F_v[i]()());
      T_v[i]()(2, 3) = timesMinusI(F_v[i]()());
      T_v[i]()(3, 2) = timesMinusI(F_v[i]()());
    });

    return T;
  }


  CloverOriginalField fillCloverXZ(const GaugeLinkField& F) {
    // NOTE: code copied from original clover term
    CloverOriginalField T(F.Grid());
    T = Zero();

    autoView(T_v, T, AcceleratorWrite);
    autoView(F_v, F, AcceleratorRead);
    accelerator_for(i, T.Grid()->oSites(), 1, {
      T_v[i]()(0, 1) = -F_v[i]()();
      T_v[i]()(1, 0) =  F_v[i]()();
      T_v[i]()(2, 3) = -F_v[i]()();
      T_v[i]()(3, 2) =  F_v[i]()();
    });

    return T;
  }


  CloverOriginalField fillCloverXY(const GaugeLinkField& F) {
    // NOTE: code copied from original clover term
    CloverOriginalField T(F.Grid());
    T = Zero();

    autoView(T_v, T, AcceleratorWrite);
    autoView(F_v, F, AcceleratorRead);
    accelerator_for(i, T.Grid()->oSites(), 1, {
      T_v[i]()(0, 0) = timesMinusI(F_v[i]()());
      T_v[i]()(1, 1) = timesI(F_v[i]()());
      T_v[i]()(2, 2) = timesMinusI(F_v[i]()());
      T_v[i]()(3, 3) = timesI(F_v[i]()());
    });

    return T;
  }


  CloverOriginalField fillCloverXT(const GaugeLinkField& F) {
    // NOTE: code copied from original clover term
    CloverOriginalField T(F.Grid());
    T = Zero();

    autoView(T_v, T, AcceleratorWrite);
    autoView(F_v, F, AcceleratorRead);
    accelerator_for(i, T.Grid()->oSites(), 1, {
      T_v[i]()(0, 1) = timesI(F_v[i]()());
      T_v[i]()(1, 0) = timesI(F_v[i]()());
      T_v[i]()(2, 3) = timesMinusI(F_v[i]()());
      T_v[i]()(3, 2) = timesMinusI(F_v[i]()());
    });

    return T;
  }


  CloverOriginalField fillCloverYT(const GaugeLinkField& F) {
    // NOTE: code copied from original clover term
    CloverOriginalField T(F.Grid());
    T = Zero();

    autoView(T_v, T, AcceleratorWrite);
    autoView(F_v, F, AcceleratorRead);
    accelerator_for(i, T.Grid()->oSites(), 1, {
      T_v[i]()(0, 1) = -F_v[i]()();
      T_v[i]()(1, 0) =  F_v[i]()();
      T_v[i]()(2, 3) =  F_v[i]()();
      T_v[i]()(3, 2) = -F_v[i]()();
    });

    return T;
  }


  CloverOriginalField fillCloverZT(const GaugeLinkField& F) {
    // NOTE: code copied from original clover term
    CloverOriginalField T(F.Grid());
    T = Zero();

    autoView(T_v, T, AcceleratorWrite);
    autoView(F_v, F, AcceleratorRead);
    accelerator_for(i, T.Grid()->oSites(), 1, {
      T_v[i]()(0, 0) = timesI(F_v[i]()());
      T_v[i]()(1, 1) = timesMinusI(F_v[i]()());
      T_v[i]()(2, 2) = timesMinusI(F_v[i]()());
      T_v[i]()(3, 3) = timesI(F_v[i]()());
    });

    return T;
  }


  GaugeLinkField Cmunu(std::vector<GaugeLinkField>& U, GaugeLinkField& lambda, int mu, int nu) {
    // NOTE: code copied from original clover term
    // Computing C_{\mu \nu}(x) as in Eq.(B.39) in Zbigniew Sroczynski's PhD thesis
    conformable(lambda.Grid(), U[0].Grid());
    GaugeLinkField out(lambda.Grid()), tmp(lambda.Grid());
    // insertion in upper staple
    // please check redundancy of shift operations

    // C1+
    tmp = lambda * U[nu];
    out = Impl::ShiftStaple(Impl::CovShiftForward(tmp, nu, Impl::CovShiftBackward(U[mu], mu, Impl::CovShiftIdentityBackward(U[nu], nu))), mu);

    // C2+
    tmp = U[mu] * Impl::ShiftStaple(adj(lambda), mu);
    out += Impl::ShiftStaple(Impl::CovShiftForward(U[nu], nu, Impl::CovShiftBackward(tmp, mu, Impl::CovShiftIdentityBackward(U[nu], nu))), mu);

    // C3+
    tmp = U[nu] * Impl::ShiftStaple(adj(lambda), nu);
    out += Impl::ShiftStaple(Impl::CovShiftForward(U[nu], nu, Impl::CovShiftBackward(U[mu], mu, Impl::CovShiftIdentityBackward(tmp, nu))), mu);

    // C4+
    out += Impl::ShiftStaple(Impl::CovShiftForward(U[nu], nu, Impl::CovShiftBackward(U[mu], mu, Impl::CovShiftIdentityBackward(U[nu], nu))), mu) * lambda;

    // insertion in lower staple
    // C1-
    out -= Impl::ShiftStaple(lambda, mu) * Impl::ShiftStaple(Impl::CovShiftBackward(U[nu], nu, Impl::CovShiftBackward(U[mu], mu, U[nu])), mu);

    // C2-
    tmp = adj(lambda) * U[nu];
    out -= Impl::ShiftStaple(Impl::CovShiftBackward(tmp, nu, Impl::CovShiftBackward(U[mu], mu, U[nu])), mu);

    // C3-
    tmp = lambda * U[nu];
    out -= Impl::ShiftStaple(Impl::CovShiftBackward(U[nu], nu, Impl::CovShiftBackward(U[mu], mu, tmp)), mu);

    // C4-
    out -= Impl::ShiftStaple(Impl::CovShiftBackward(U[nu], nu, Impl::CovShiftBackward(U[mu], mu, U[nu])), mu) * lambda;

    return out;
  }

  /////////////////////////////////////////////
  // Helpers
  /////////////////////////////////////////////

private:

  template<typename vobj>
  accelerator_inline vobj triangle_elem(const iImplCloverTriangle<vobj>& triangle, int block, int i, int j) {
    assert(i != j);
    if(i < j) {
      return triangle()(block)(triangle_index(i, j));
    } else { // i > j
      return conjugate(triangle()(block)(triangle_index(i, j)));
    }
  }


  accelerator_inline int triangle_index(int i, int j) {
    if(i == j)
      return 0;
    else if(i < j)
      return Nred * (Nred - 1) / 2 - (Nred - i) * (Nred - i - 1) / 2 + j - i - 1;
    else // i > j
      return Nred * (Nred - 1) / 2 - (Nred - j) * (Nred - j - 1) / 2 + i - j - 1;
  }

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

private:

  RealD csw_r;
  RealD csw_t;
  RealD cF;

  bool open_boundaries;

  CloverDiagonalField Diagonal,    DiagonalEven,    DiagonalOdd;
  CloverDiagonalField DiagonalInv, DiagonalInvEven, DiagonalInvOdd;

  CloverTriangleField Triangle,    TriangleEven,    TriangleOdd;
  CloverTriangleField TriangleInv, TriangleInvEven, TriangleInvOdd;

  FermionField Tmp;

  MaskField BoundaryMask, BoundaryMaskEven, BoundaryMaskOdd;
};
