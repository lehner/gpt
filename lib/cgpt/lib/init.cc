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
#define _THIS_IS_INIT_ // needed for numpy array
#include "lib.h"

static bool cgpt_initialized = false;

EXPORT(init,{

    PyObject* _args;
    if (!PyArg_ParseTuple(args, "O", &_args)) {
      return NULL;
    }
    
    std::vector<std::string> sargs;
    cgpt_convert(_args,sargs);
    
    // make cargs
    std::vector<char*> cargs;
    for (auto& a : sargs) {
      cargs.push_back((char*)a.c_str());
    }
    cargs.push_back(0);
    
    int argc = (int)sargs.size();
    char** argv = &cargs[0];

    // initialize Grid
    Grid_init(&argc,&argv);

    // initialize numpy as well
    import_array();
    
    std::cout << std::endl <<
      "=============================================" << std::endl <<
      "              Initialized GPT                " << std::endl <<
      "    Copyright (C) 2020 Christoph Lehner      " << std::endl <<
      "=============================================" << std::endl;

    cgpt_initialized = true;

    // Initial test of new global memory system
    // test_global_memory_system();     
    // Grid_finalize();
    // exit(0);
    
    return PyLong_FromLong(0);
    
  });


EXPORT(exit,{

    if (cgpt_initialized) {

      std::cout <<
	"=============================================" << std::endl <<
	"               Finalized GPT                 " << std::endl <<
	"=============================================" << std::endl;

      Grid_finalize();
      cgpt_initialized = false;

    }


    return PyLong_FromLong(0);
    
  });

