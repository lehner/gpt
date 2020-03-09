/*
  CGPT

  Authors: Christoph Lehner 2020
*/
EXPORT_BEGIN(init) {
       
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
  
  return PyLong_FromLong(0);
  
} EXPORT_END();
