/*
  CGPT

  Authors: Christoph Lehner 2020
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

static PyObject* load_nersc(PyObject* args) { 

  ASSERT(PyTuple_Check(args));

  if (PyTuple_Size(args) == 2) {

    std::string filename;
    bool verbose;

    cgpt_convert(PyTuple_GetItem(args,0),filename);
    cgpt_convert(PyTuple_GetItem(args,1),verbose);

    // get metadata
    std::map<std::string,std::string> fields;
    if (!read_nersc_header(filename,fields)) {
      if (verbose)
	std::cout << GridLogMessage << "GPT::IO: read nersc header failed" << std::endl;
      return NULL;
    }

    std::vector<int> gdimension(4);
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
      SpaceTimeGrid::makeFourDimGrid(gdimension, GridDefaultSimd(4,vComplexD::Nsimd()), GridDefaultMpi());

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
  }

  return NULL;
}
