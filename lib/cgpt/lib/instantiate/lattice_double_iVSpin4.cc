// This file is automatically generated, do not modify!
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
#include "../lib.h"

typedef void* (* create_lattice_prec_otype)(GridBase* grid);
extern std::map<std::string,create_lattice_prec_otype> _create_otype_;
extern std::map<std::string,int> _otype_singlet_rank_;

// explicitly instantiate
template class cgpt_Lattice<iVSpin4<vComplexD>>;

void lattice_init_double_iVSpin4() {
  std::string prec = "double";
  _create_otype_[prec + ":" + get_otype(iVSpin4<vComplexD>())] = [](GridBase* grid) { return (void*)new cgpt_Lattice< iVSpin4< vComplexD > >(grid); };
  _otype_singlet_rank_[get_otype(iVSpin4<vComplexD>())] = singlet_rank(iVSpin4<vComplexD>());
}
