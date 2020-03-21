/*
  CGPT

  Authors: Christoph Lehner 2020
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

