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
static bool cgpt_quiet = false;

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

    // quiet mode
    cgpt_quiet = getenv("GPT_QUIET") != 0;
    std::streambuf* rb_prev = std::cout.rdbuf();
    std::ostringstream rb_str;
    if (cgpt_quiet) {
      std::cout.rdbuf( rb_str.rdbuf() );
    }
    
    // initialize Grid
    Grid_init(&argc, &argv);

    // initialize numpy as well
    import_array();

    std::cout << std::endl <<
      "=============================================" << std::endl <<
      "              Initialized GPT                " << std::endl <<
      "     Copyright (C) 2020 Christoph Lehner     " << std::endl <<
      "=============================================" << std::endl;
    
    cgpt_initialized = true;

    if (cgpt_quiet) {
      std::cout.rdbuf( rb_prev );
    }

    return PyLong_FromLong(0);
    
  });


EXPORT(exit,{

    if (cgpt_initialized) {

      if (!cgpt_quiet) {
	std::cout <<
	  "=============================================" << std::endl <<
	  "               Finalized GPT                 " << std::endl <<
	  "=============================================" << std::endl;
      }

      if (getenv("GPT_SUPPRESS_GRID_FINALIZE") == 0)
	Grid_finalize();
      
      cgpt_initialized = false;

    }


    return PyLong_FromLong(0);
    
  });

