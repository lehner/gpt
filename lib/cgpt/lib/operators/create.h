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
template<typename vCoeff_t>
cgpt_fermion_operator_base* cgpt_create_fermion_operator(const std::string& optype, PyObject* args) {

  if (optype == "wilson_clover") {
    return cgpt_create_wilson_clover<vCoeff_t>(args);
  } else if (optype == "zmobius") {
    return cgpt_create_zmobius<vCoeff_t>(args);
  } else if (optype == "mobius") {
    return cgpt_create_mobius<vCoeff_t>(args);
  } else if (optype == "coarse") {
    return cgpt_create_coarsenedmatrix<vCoeff_t>(args);
  } else if (optype == "wilson_twisted_mass") {
    return cgpt_create_wilson_twisted_mass<vCoeff_t>(args);
  } else {
    ERR("Unknown operator type");
  }

}
