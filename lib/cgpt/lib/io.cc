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
#include "lib.h"
#include "io/nersc.h"
#include "io/openQCD.h"

EXPORT(load,{
    PyObject* ret;

    bool verbose;
    cgpt_convert(PyTuple_GetItem(args,1),verbose);

    if ((ret = load_nersc(args)))
      return ret;

    // openQCD file format is minimal, not distinctive, test last
    if ((ret = load_openQCD(args)))
      return ret;

    ERR("Unknown file format!");
    
    Py_RETURN_NONE;
    
  });

EXPORT(save,{

    std::string dest, format;
    bool verbose;
    PyObject* _dest,* _objs,* _format,* _verbose;
    if (!PyArg_ParseTuple(args, "OOOO", &_dest, &_objs, &_format, &_verbose)) {
      return NULL;
    }

    cgpt_convert(_dest,dest);
    cgpt_convert(_format,format);
    cgpt_convert(_verbose,verbose);

    //if (format == "gpt") {
    //  return save_gpt(dest,_objs,verbose);
    //}

    ERR("Unknown format: %s", format.c_str());
    
    Py_RETURN_NONE;
    
  });
