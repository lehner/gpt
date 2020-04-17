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
#include "random/distribution.h"
#include "random/engine.h"

/*
  rng[hash[pos]]

  hash[pos] : divide lattice into 2^nd blocks/hashes
              need to make sure that no two ranks have same hash

	      for 32^4 local volume, this means 65536 local hashes

	      when do we send all hashes do root to check for duplicates
*/

EXPORT(create_random,{

    PyObject* _type;
    std::string type;
    if (!PyArg_ParseTuple(args, "O", &_type)) {
      return NULL;
    }
    cgpt_convert(_type,type);

    if (type == "ranlux48") {
      return PyLong_FromVoidPtr(new cgpt_random_engine<std::ranlux48>());
    } else {
      ERR("Unknown rng engine type %s",type.c_str());
    }

    return PyLong_FromLong(0);
    
  });

EXPORT(delete_random,{

    void* _p;
    if (!PyArg_ParseTuple(args,"l", &_p)) {
      return NULL;
    }

    cgpt_random_engine_base* p = (cgpt_random_engine_base*)_p;
    delete p;

    return PyLong_FromLong(0);
  });

EXPORT(random_seed,{

    PyObject* _seed_str;
    void* _p;
    if (!PyArg_ParseTuple(args, "lO", &_p,&_seed_str)) {
      return NULL;
    }

    std::string seed_str;
    cgpt_convert(_seed_str,seed_str);

    cgpt_random_engine_base* p = (cgpt_random_engine_base*)_p;
    p->seed(seed_str);

    return PyLong_FromLong(0);

  });

EXPORT(random_sample,{

    PyObject* _target, *_param;
    void* _p;
    if (!PyArg_ParseTuple(args, "lOO", &_p,&_target,&_param)) {
      return NULL;
    }

    cgpt_random_engine_base* p = (cgpt_random_engine_base*)_p;
    return p->sample(_target,_param);

  });
