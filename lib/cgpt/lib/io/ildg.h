/*
    GPT - Grid Python Toolkit
    Copyright (C) 2024  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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
static PyObject* load_ildg(PyObject* args) { 

  ASSERT(PyTuple_Check(args));

  if (PyTuple_Size(args) == 2) {

    std::string filename;
    bool verbose;

    cgpt_convert(PyTuple_GetItem(args,0),filename);
    cgpt_convert(PyTuple_GetItem(args,1),verbose);

    // check is file
    if (!cgpt_is_file(filename))
      return NULL;

    // check if it is in ILDG format
    const uint32_t ILDG_MAGIC = 0xAB896745;
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f)
      return NULL;
    uint32_t FILE_MAGIC = 0x0;
    fread(&FILE_MAGIC, 4, 1, f);
    fclose(f);

    if (FILE_MAGIC != ILDG_MAGIC)
      return NULL;

#ifndef HAVE_LIME
    std::cout << GridLogMessage << "Warning: found an ILDG configuration but did not compile Grid with LIME support" << std::endl;
#else
    
    // now get metadata
    IldgReader reader;
    reader.open(filename);
    std::string xml;
    
    reader.readLimeObject(xml, ILDG_FORMAT);
    reader.close();

    XmlReader RD(xml, true, "");
    ildgFormat ildgFormat;
    read(RD,"ildgFormat",ildgFormat);

    if (ildgFormat.precision != 64) {
      std::cout << GridLogMessage << "Warning: GPT for now only supports double-precision ILDG configurations" << std::endl;
      return NULL;
    }

    // construct Grid
    std::vector<int> gdimension(4), cb_mask(4,0);
    gdimension[0] = ildgFormat.lx;
    gdimension[1] = ildgFormat.ly;
    gdimension[2] = ildgFormat.lz;
    gdimension[3] = ildgFormat.lt;
    GridCartesian* grid = 
      (GridCartesian*)cgpt_create_grid(cgpt_to_coordinate(gdimension),GridDefaultSimd(4,vComplexD::Nsimd()), cb_mask, GridDefaultMpi(),0);

    // load gauge field
    LatticeGaugeFieldD Umu(grid);

    FieldMetaData header;
    reader.open(filename);
    reader.readConfiguration(Umu,header);
    reader.close();
  
    std::vector< cgpt_Lattice_base* > U(4);
    for (int mu=0;mu<4;mu++) {
      auto lat = new cgpt_Lattice< iColourMatrix< vComplexD > >(grid);
      lat->l = PeekIndex<LorentzIndex>(Umu,mu);
      U[mu] = lat;
    }

    // build metadata
    PyObject* metadata = PyDict_New();
    //for (auto& k : fields) {
    //  PyObject* data = PyUnicode_FromString(k.second.c_str());
    //  PyDict_SetItemString(metadata,k.first.c_str(),data); Py_XDECREF(data);
    //}

    // return
    vComplexD vScalar = 0; // TODO: grid->to_decl()
    return Py_BuildValue("([(l,[i,i,i,i],s,s,[N,N,N,N])],N)", grid, gdimension[0], gdimension[1], gdimension[2],
			 gdimension[3], get_prec(vScalar).c_str(), "full", U[0]->to_decl(), U[1]->to_decl(), U[2]->to_decl(),
			 U[3]->to_decl(),metadata);

#endif
  }
  return NULL;
}
