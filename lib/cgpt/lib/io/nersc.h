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
static bool read_nersc_header(std::string filename, std::map<std::string,std::string>& fields) {
  // use Grid's removeWhitespace
  std::string ln;
  std::ifstream f(filename);
  getline(f,ln); removeWhitespace(ln);
  if (ln != "BEGIN_HEADER")
    return false;
  do {
    getline(f,ln); removeWhitespace(ln);
    int i = ln.find("=");
    if(i>0) {
      auto k=ln.substr(0,i); removeWhitespace(k);
      auto v=ln.substr(i+1); removeWhitespace(v);
      fields[k] = v;
    }
  } while( ln != "END_HEADER" );
  return true;
}

static void save_nersc(const std::string& filename,
		       PyObject* format,
		       PyObject* objs,
		       bool verbose) {

  std::vector< cgpt_Lattice_base* > U;
  long npl = cgpt_basis_fill(U, objs);

  PyObject* params = PyObject_GetAttrString(format, "params");
  ASSERT(params);
  std::string label = get_str(params, "label");
  std::string id = get_str(params, "id");
  int sequence_number = get_int(params, "sequence_number");
  Py_DECREF(params);
  
  ASSERT(npl == 1);
  ASSERT(U.size() == 4);

  LatticeGaugeFieldD Umu(U[0]->get_grid());
  
  for (int mu=0;mu<4;mu++) {
    auto lat = compatible< iColourMatrix< vComplexD > >(U[mu]);
    PokeIndex<LorentzIndex>(Umu,lat->l,mu);
  }

  NerscIO::writeConfiguration(Umu,filename,0,0,
			      label,id,(unsigned int)sequence_number);

}

static PyObject* load_nersc(PyObject* args) { 

  ASSERT(PyTuple_Check(args));

  if (PyTuple_Size(args) == 2) {

    std::string filename;
    bool verbose;

    cgpt_convert(PyTuple_GetItem(args,0),filename);
    cgpt_convert(PyTuple_GetItem(args,1),verbose);

    // check is file
    if (!cgpt_is_file(filename))
      return NULL;

    // get metadata
    std::map<std::string,std::string> fields;
    if (!read_nersc_header(filename,fields)) {
      return NULL;
    }

    std::vector<int> gdimension(4), cb_mask(4,0);
    for (int i=0;i<4;i++) {
      char buf[32];
      sprintf(buf,"DIMENSION_%d",i+1);
      gdimension[i] = atoi(fields[buf].c_str());
      if (verbose)
	std::cout << GridLogMessage << "GPT::IO: gdimension[" << i << "] = " << gdimension[i] << std::endl;
    }

    // construct Grid
    assert(Nd == 4);
    GridCartesian* grid = 
      (GridCartesian*)cgpt_create_grid(cgpt_to_coordinate(gdimension),GridDefaultSimd(4,vComplexD::Nsimd()), cb_mask, GridDefaultMpi(),0);

    // load gauge field
    LatticeGaugeFieldD Umu(grid);

    FieldMetaData header;
    NerscIO::readConfiguration(Umu,header,filename);

    std::vector< cgpt_Lattice_base* > U(4);
    for (int mu=0;mu<4;mu++) {
      auto lat = new cgpt_Lattice< iColourMatrix< vComplexD > >(grid);
      lat->l = PeekIndex<LorentzIndex>(Umu,mu);
      U[mu] = lat;
    }

    // build metadata
    PyObject* metadata = PyDict_New();
    for (auto& k : fields) {
      PyObject* data = PyUnicode_FromString(k.second.c_str());
      PyDict_SetItemString(metadata,k.first.c_str(),data); Py_XDECREF(data);
    }

    // return
    vComplexD vScalar = 0; // TODO: grid->to_decl()
    return Py_BuildValue("([(l,[i,i,i,i],s,s,[N,N,N,N])],N)", grid, gdimension[0], gdimension[1], gdimension[2],
			 gdimension[3], get_prec(vScalar).c_str(), "full", U[0]->to_decl(), U[1]->to_decl(), U[2]->to_decl(),
			 U[3]->to_decl(),metadata);
  }

  return NULL;
}
