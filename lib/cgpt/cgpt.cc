#include <Python.h>
#include <vector>
#include <string>
#include <iostream>

#include <Grid/Grid.h>

using namespace Grid;

#include "lib/convert.h"
#include "lib/grid.h"
#include "lib/peekpoke.h"
#include "lib/lattice.h"
#include "lib/init.h"
#include "lib/mpi.h"

static PyMethodDef module_functions[] = {
  {"init", cgpt_init, METH_VARARGS, "Initializes gpt"},
  {"create_grid", cgpt_create_grid, METH_VARARGS, "Creates a grid"},
  {"delete_grid", cgpt_delete_grid, METH_VARARGS, "Deletes a grid"},
  {"grid_barrier", cgpt_grid_barrier, METH_VARARGS, "Grid::Barrier"},
  {"create_lattice", cgpt_create_lattice, METH_VARARGS, "Creates a lattice"},
  {"delete_lattice", cgpt_delete_lattice, METH_VARARGS, "Deletes a lattice"},
  {"lattice_set_val", cgpt_lattice_set_val, METH_VARARGS, "Set a value within a lattice"},
  {"lattice_to_str", cgpt_lattice_to_str, METH_VARARGS, "Get a string representation of the lattice"},
  {"global_rank", cgpt_global_rank, METH_VARARGS, "Rank within global MPI"},
  {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef module_def = {
  PyModuleDef_HEAD_INIT,
  "cgpt",     /* m_name */
  "The C++ interface between gpt and Grid",  /* m_doc */
  -1,                  /* m_size */
  module_functions,    /* m_methods */
  NULL,                /* m_reload */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  NULL,                /* m_free */
};

// Export module creation
PyMODINIT_FUNC PyInit_cgpt(void){
  PyModule_Create(&module_def);
}
