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
void lattice_init();

// type name shortcuts
#define BASIS_SIZE(n) \
  template<typename vtype> using iVSinglet ## n = iVector<iSinglet<vtype>,n>; \
  template<typename vtype> using iMSinglet ## n = iMatrix<iSinglet<vtype>,n>;
#include "../basis_size.h"
#undef BASIS_SIZE

#define SPIN_COLOR(Ns,Nc)						\
  template<typename vtype> using iMSpin ## Ns ## Color ## Nc = iScalar<iMatrix<iMatrix<vtype, Nc>, Ns> >; \
  template<typename vtype> using iVSpin ## Ns ## Color ## Nc = iScalar<iVector<iVector<vtype, Nc>, Ns> >;

#define SPIN(Ns)							\
  template<typename vtype> using iMSpin ## Ns = iScalar<iMatrix<iScalar<vtype>, Ns> >; \
  template<typename vtype> using iVSpin ## Ns = iScalar<iVector<iScalar<vtype>, Ns> >;

#define COLOR(Nc)							\
  template<typename vtype> using iMColor ## Nc = iScalar<iScalar<iMatrix<vtype, Nc> > > ; \
  template<typename vtype> using iVColor ## Nc = iScalar<iScalar<iVector<vtype, Nc> > > ;

#include "../spin_color.h"

#undef SPIN_COLOR
#undef SPIN
#undef COLOR

// map types to strings
static const std::string get_prec(const vComplexF& l) { return "single"; };
static const std::string get_prec(const vComplexD& l) { return "double"; };
template<typename T> const std::string get_prec(const Lattice<T>& l) { typedef typename Lattice<T>::vector_type vCoeff_t; vCoeff_t t; return get_prec(t); }

template<typename vtype> const std::string get_otype(const iSinglet<vtype>& l) { return "ot_singlet"; };
template<typename vtype, int Nc> const std::string get_otype(const iScalar<iScalar<iMatrix<vtype, Nc> > >& l) { return "ot_mcolor" + std::to_string(Nc); };
template<typename vtype, int Nc> const std::string get_otype(const iScalar<iScalar<iVector<vtype, Nc> > >& l) { return "ot_vcolor" + std::to_string(Nc); };
template<typename vtype, int Ns> const std::string get_otype(const iScalar<iMatrix<iScalar<vtype>, Ns> >& l) { return "ot_mspin" + std::to_string(Ns); };
template<typename vtype, int Ns> const std::string get_otype(const iScalar<iVector<iScalar<vtype>, Ns> >& l) { return "ot_vspin" + std::to_string(Ns); };
template<typename vtype, int Ns, int Nc> const std::string get_otype(const iScalar<iMatrix<iMatrix<vtype, Nc>, Ns> >& l) { return "ot_mspin" + std::to_string(Ns) + "color" + std::to_string(Nc); };
template<typename vtype, int Ns, int Nc> const std::string get_otype(const iScalar<iVector<iVector<vtype, Nc>, Ns> >& l) { return "ot_vspin" + std::to_string(Ns) + "color" + std::to_string(Nc); };
template<typename vtype,int nbasis> const std::string get_otype(const iVector<iSinglet<vtype>,nbasis>& l) { return std::string("ot_vsinglet") + std::to_string(nbasis); };
template<typename vtype,int nbasis> const std::string get_otype(const iMatrix<iSinglet<vtype>,nbasis>& l) { return std::string("ot_msinglet") + std::to_string(nbasis); };
template<typename T> const std::string get_otype(const Lattice<T>& l) { typedef typename Lattice<T>::vector_object vobj; vobj t; return get_otype(t); }
