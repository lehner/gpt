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
static bool read_openQCD_header(std::string filename, std::map<std::string,std::string>& fields) {
  FILE* f = fopen(filename.c_str(),"rb");
  if (!f)
    return false;
  OpenQcdHeader header;
  if (fread(&header,sizeof(header),1,f)!=1)
    return false;
  char buf[64];

  // since header is minimal, decide if this is openQCD file by very weak standards
#define CHECK_EXTENT(L) if (L<1 || L > 10000)return false;
  CHECK_EXTENT(header.Nx);
  CHECK_EXTENT(header.Ny);
  CHECK_EXTENT(header.Nz);
  CHECK_EXTENT(header.Nt);
#undef CHECK_EXTENT

  if (header.plaq < -100.0 || header.plaq > 100.0)
    return false;

  // keep metadata
  sprintf(buf,"%d",header.Nx); 
  fields["DIMENSION_1"] = buf;

  sprintf(buf,"%d",header.Ny);
  fields["DIMENSION_2"] = buf;

  sprintf(buf,"%d",header.Nz);
  fields["DIMENSION_3"] = buf;

  sprintf(buf,"%d",header.Nt);
  fields["DIMENSION_4"] = buf;

  sprintf(buf,"%.15f",header.plaq);
  fields["PLAQUETTE"] = buf;


  return true;
}

static PyObject* load_openQCD(PyObject* args) { 

  ASSERT(PyTuple_Check(args));

  if (PyTuple_Size(args) == 2) {

    std::string filename;
    bool verbose;

    cgpt_convert(PyTuple_GetItem(args,0),filename);
    cgpt_convert(PyTuple_GetItem(args,1),verbose);

    // get metadata
    std::map<std::string,std::string> fields;
    if (!read_openQCD_header(filename,fields)) {
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
    OpenQcdIO::readConfiguration(Umu,header,filename);

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
    vComplexD vScalar = 0;
    return Py_BuildValue("([(l,[i,i,i,i],s,s,[N,N,N,N])],N)", grid, gdimension[0], gdimension[1], gdimension[2],
			 gdimension[3], get_prec(vScalar).c_str(), "full", U[0]->to_decl(), U[1]->to_decl(), U[2]->to_decl(),
			 U[3]->to_decl(),metadata);
  }

  return NULL;
}
