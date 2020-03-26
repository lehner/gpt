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
static const char* get_prec(const vComplexF& l) { return "single"; };
static const char* get_prec(const vComplexD& l) { return "double"; };

template<typename T> const char* get_prec(const Lattice<T>& l) { typedef typename Lattice<T>::vector_type vCoeff_t; vCoeff_t t; return get_prec(t); }
template<typename T> const char* get_otype(const Lattice<T>& l) { typedef typename Lattice<T>::vector_object vobj; vobj t; return get_otype(t); }

template<typename vobj> const char* get_otype(const iSinglet<vobj>& l) { return "ot_complex"; };
template<typename vobj> const char* get_otype(const iColourMatrix<vobj>& l) { return "ot_mcolor"; };
template<typename vobj> const char* get_otype(const iColourVector<vobj>& l) { return "ot_vcolor"; };
template<typename vobj> const char* get_otype(const iSpinColourMatrix<vobj>& l) { return "ot_mspincolor"; };
template<typename vobj> const char* get_otype(const iSpinColourVector<vobj>& l) { return "ot_vspincolor"; };
