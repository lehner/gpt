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

/*

The following comments are parsed by ./make and
generate the mapping of function names to op codes.

BEGIN_EXPORT_UNARY_REALD
END_EXPORT_UNARY_REALD

BEGIN_EXPORT_UNARY_VOID
M
Mdag
Meooe
MeooeDag
Mooee
MooeeDag
MooeeInv
MooeeInvDag
Mdiag
Dminus
DminusDag
ImportPhysicalFermionSource
ImportUnphysicalFermion
ExportPhysicalFermionSolution
ExportPhysicalFermionSource
END_EXPORT_UNARY_VOID

BEGIN_EXPORT_UNARY_DAG_VOID
Dhop
DhopEO
END_EXPORT_UNARY_DAG_VOID

BEGIN_EXPORT_DERIV_DAG_VOID
MDeriv
MoeDeriv
MeoDeriv
DhopDeriv
DhopDerivEO
DhopDerivOE
END_EXPORT_DERIV_DAG_VOID

BEGIN_EXPORT_DIRDISP_VOID
Mdir
END_EXPORT_DIRDISP_VOID

*/
